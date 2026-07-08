import torch
import torch.nn as nn

from models.image_model import ImageModel
from models.clinical_model import ClinicalModel
from models.geo_model import GeoModel

class FusionModel(nn.Module):

    def __init__(self, clinical_input_dim):

        super().__init__()

        self.image_model = ImageModel()

        self.clinical_model = ClinicalModel(clinical_input_dim)

        self.geo_model = GeoModel()

        # REMOVE IMAGE CLASSIFIER
        self.image_model.classifier = nn.Identity()

        # FINAL FUSION
        self.fusion = nn.Sequential(

            nn.Linear(1280 + 64 + 32, 256),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(256, 1)
        )

    def forward(self, image, clinical, geo):

        image_features = self.image_model.backbone(image)

        clinical_features = self.clinical_model(clinical)

        geo_features = self.geo_model(geo)

        combined = torch.cat(
            [image_features, clinical_features, geo_features],
            dim=1
        )

        output = self.fusion(combined)

        return output