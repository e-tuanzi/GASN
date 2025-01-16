from model.pddn import PromptDrivenDensityNetwork
from model.famnet import FamNet
from par_segment_anything import SamPredictor, sam_model_registry
from par_segment_anything.automatic_mask_generator import SamAutomaticMaskGenerator
from par_segment_anything.utils.transforms import FSCTransform


class Gasn:
    def __init__(self, dmg_path, sam_type, sam_path, type="pddn", device="cuda"):
        self.trans = FSCTransform()
        if type == "pddn":
            self.pddn = PromptDrivenDensityNetwork(dmg_path, device)
        elif type == "famnet":
            self.famnet = FamNet(dmg_path, device)
        self.sam = sam_model_registry[sam_type](checkpoint=sam_path)
        self.sam.to(device=device)
        self.sam_predictor = SamPredictor(self.sam)

    def predict(self, image, boxes):
        self.sam_predictor.set_image(image)
        image_embedding = self.sam_predictor.get_image_embedding()
        output = self.pddn.predict(image_embedding, boxes)
        return output

    def _train_predict(self, image_embedding, boxes):
        output = self.pddn.predict(image_embedding, boxes)
        return output

    def _famnet_predict(self, image, boxes):
        output = self.famnet.predict(image, boxes)
        output = output.squeeze().detach().cpu().numpy()
        output = self.trans.resize_gt(output).unsqueeze(0)
        return output
