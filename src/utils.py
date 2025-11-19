# src/data_augmentation.py

def augment_video_features(appearance, motion, aug_prob=0.3):
    if random.random() < aug_prob:
        # Temporal masking
        t_app, d_app = appearance.shape
        t_mot, d_mot = motion.shape
        
        mask_len_app = max(1, int(t_app * 0.1))
        mask_len_mot = max(1, int(t_mot * 0.1))
        
        start_app = random.randint(0, t_app - mask_len_app)
        start_mot = random.randint(0, t_mot - mask_len_mot)
        
        appearance[start_app:start_app + mask_len_app] = 0
        motion[start_mot:start_mot + mask_len_mot] = 0
    
    return appearance, motion

# src/ensemble.py
class EnsembleModel:
    def __init__(self, model_paths):
        self.models = []
        for path in model_paths:
            model = EarlyFusionQA(text_model_name="vinai/phobert-base")
            model.load_state_dict(torch.load(path)["model"])
            model.eval()
            self.models.append(model)
    
    def predict(self, input_ids, attention_mask, appearance, motion):
        all_logits = []
        for model in self.models:
            with torch.no_grad():
                logits = model(input_ids, attention_mask, appearance, motion)
                all_logits.append(logits)
        
        avg_logits = torch.mean(torch.stack(all_logits), dim=0)
        return avg_logits
    
