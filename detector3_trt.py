# local YOLO detector

import cv2
import numpy as np
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda

TRT_LOGGER = trt.Logger()
INPUT_W = 640
INPUT_H = 640
CONF_THRESH = 0.5
IOU_THRESHOLD = 0.4

# load coco labels
categories = ["person", "bicycle", "car"]


class Detector3:

    def __init__(self, engine_file_path):
        self.img_size = 640
        self.threshold = CONF_THRESH
        self.stride = 1

        # Create a Context on this device,
        self.cfx = cuda.Device(1).make_context()
        stream = cuda.Stream()
        TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
        runtime = trt.Runtime(TRT_LOGGER)

        # Deserialize the engine from file
        with open(engine_file_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()

        host_inputs = []
        cuda_inputs = []
        host_outputs = []
        cuda_outputs = []
        bindings = []

        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(cuda_mem))
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)
            else:
                host_outputs.append(host_mem)
                cuda_outputs.append(cuda_mem)

        # Store
        self.stream = stream
        self.context = context
        self.engine = engine
        self.host_inputs = host_inputs
        self.cuda_inputs = cuda_inputs
        self.host_outputs = host_outputs
        self.cuda_outputs = cuda_outputs
        self.bindings = bindings

    def detect(self, im, x_prev, y_prev):
    # def detect(self, im):
        # threading.Thread.__init__(self)
        # Make self the active context, pushing it on top of the context stack.
        self.cfx.push()
        # Restore
        stream = self.stream
        context = self.context
        engine = self.engine
        host_inputs = self.host_inputs
        cuda_inputs = self.cuda_inputs
        host_outputs = self.host_outputs
        cuda_outputs = self.cuda_outputs
        bindings = self.bindings
        # Do image preprocess
        input_image, image_raw, origin_h, origin_w = self.preprocess_image(im)
        # Copy input image to host buffer
        np.copyto(host_inputs[0], input_image.ravel())
        # Transfer input data  to the GPU.
        cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
        # Run inference.
        context.execute_async(bindings=bindings, stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
        # Synchronize the stream
        stream.synchronize()
        # Remove any context from the top of the context stack, deactivating it.
        # self.cfx.pop()
        # Here we use the first row of output in that batch_size = 1
        trt_outputs = host_outputs[0]
        # Do postprocess
        results_trt = self.post_process(trt_outputs, origin_h, origin_w, x_prev, y_prev)
        # results_trt = self.post_process(trt_outputs, origin_h, origin_w)

        return results_trt

    def preprocess_image(self, image_raw):
        """
        description: Read an image from image path, convert it to RGB,
                     resize and pad it to target size, normalize to [0,1],
                     transform to NCHW format.
        param:
            input_image_path: str, image path
        return:
            image:  the processed image
            image_raw: the original image
            h: original height
            w: original width
        """
        h, w, c = image_raw.shape
        image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
        # Calculate widht and height and paddings
        r_w = INPUT_W / w
        r_h = INPUT_H / h
        if r_h > r_w:
            tw = INPUT_W
            th = int(r_w * h)
            tx1 = tx2 = 0
            ty1 = int((INPUT_H - th) / 2)
            ty2 = INPUT_H - th - ty1
        else:
            tw = int(r_h * w)
            th = INPUT_H
            tx1 = int((INPUT_W - tw) / 2)
            tx2 = INPUT_W - tw - tx1
            ty1 = ty2 = 0
        # Resize the image with long side while maintaining ratio
        image = cv2.resize(image, (tw, th))
        # Pad the short side with (128,128,128)
        image = cv2.copyMakeBorder(
            image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, (128, 128, 128)
        )
        image = image.astype(np.float32)
        # Normalize to [0,1]
        image /= 255.0
        # HWC to CHW format:
        image = np.transpose(image, [2, 0, 1])
        # CHW to NCHW format
        image = np.expand_dims(image, axis=0)
        # Convert the image to row-major order, also known as "C order":
        image = np.ascontiguousarray(image)
        return image, image_raw, h, w

    def xywh2xyxy(self, origin_h, origin_w, x):
        """
        description:    Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        param:
            origin_h:   height of original image
            origin_w:   width of original image
            x:          A boxes tensor, each row is a box [center_x, center_y, w, h]
        return:
            y:          A boxes tensor, each row is a box [x1, y1, x2, y2]
        """
        # y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
        y = np.zeros_like(x)
        r_w = INPUT_W / origin_w
        r_h = INPUT_H / origin_h
        if r_h > r_w:
            y[:, 0] = x[:, 0] - x[:, 2] / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2 - (INPUT_H - r_w * origin_h) / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2 - (INPUT_H - r_w * origin_h) / 2
            y /= r_w
        else:
            y[:, 0] = x[:, 0] - x[:, 2] / 2 - (INPUT_W - r_h * origin_w) / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2 - (INPUT_W - r_h * origin_w) / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2
            y /= r_h

        # back to x,y,w,h format
        y[:, 2] = y[:, 2] - y[:, 0]
        y[:, 3] = y[:, 3] - y[:, 1]
        return y

    def dist(self, x1, y1, x2, y2):
        distance = np.sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1))
        return distance

    def post_process(self, output, origin_h, origin_w, x_prev, y_prev):
    # def post_process(self, output, origin_h, origin_w):
        """
        description: postprocess the prediction
        param:
            output:     A tensor likes [num_boxes,cx,cy,w,h,conf,cls_id, cx,cy,w,h,conf,cls_id, ...]
            origin_h:   height of original image
            origin_w:   width of original image
        return:
            result_boxes: finally boxes, a boxes tensor, each row is a box [x1, y1, x2, y2]
            result_scores: finally scores, a tensor, each element is the score correspoing to box
            result_classid: finally classid, a tensor, each element is the classid correspoing to box
        """
        # Get the num of boxes detected
        num = int(output[0])
        # Reshape to a two dimentional ndarray
        pred = np.reshape(output[1:], (-1, 6))[:num, :]
        # to a torch Tensor
        # pred = torch.Tensor(pred).cuda()
        # Get the boxes
        boxes = pred[:, :4]
        # Get the scores
        scores = pred[:, 4]
        # Get the classid
        classid = pred[:, 5]
        # Choose those boxes that score > CONF_THRESH
        si = scores > CONF_THRESH
        boxes = boxes[si, :]
        scores = scores[si].tolist()
        classid = classid[si].tolist()
        # Trandform bbox from [center_x, center_y, w, h] to [x1, y1, x2, y2]
        boxes = self.xywh2xyxy(origin_h, origin_w, boxes).tolist()
        # Do nms
        # indices = torchvision.ops.nms(boxes, scores, iou_threshold=IOU_THRESHOLD).cpu()
        indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=CONF_THRESH, nms_threshold=IOU_THRESHOLD)
        # print(indices)
        result_boxes = np.array(boxes)
        result_scores = np.array(scores)
        result_classid = np.array(classid)
        if len(boxes) > 0:
            result_boxes = result_boxes[indices, :]
            result_scores = result_scores[indices]
            result_classid = result_classid[indices]
        # back to x1, y1, x2, y2 format
        result_boxes = np.reshape(result_boxes, (-1, 4))
        result_boxes[:, 2] = result_boxes[:, 2] + result_boxes[:, 0]
        result_boxes[:, 3] = result_boxes[:, 3] + result_boxes[:, 1]
        #
        results_trt = []
        final_box = []
        # ref_score = 0.45
        ref_dist = 10
        for i in range(len(result_boxes)):
            x1, y1 = int(result_boxes[i][0]), int(result_boxes[i][1])
            x2, y2 = int(result_boxes[i][2]), int(result_boxes[i][3])
            cid = result_classid[i][0]
            label = categories[int(cid)]
            conf = result_scores[i][0]
            x_now = (x1 + x2) / 2
            y_now = (y1 + y2) / 2
            distance = self.dist(x_prev, y_prev, x_now, y_now)
            # print('distance is: ', distance)
            # results_trt.append([x1, y1, x2, y2, label, conf, distance])
            # if conf > ref_score and distance < ref_dist:
            #     final_box = np.array([x1, y1, x2 - x1, y2 - y1])
            #     ref_score = conf
            #     ref_dist = distance
            if distance < ref_dist:
                final_box = np.array([x1, y1, x2 - x1, y2 - y1])
                ref_dist = distance
            # if conf > ref_score:
            #     final_box = np.array([x1, y1, x2 - x1, y2 - y1])
            #     ref_score = conf

        return final_box

    def destroy(self):
        # Remove any context from the top of the context stack, deactivating it.
        self.cfx.pop()