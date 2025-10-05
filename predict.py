from cog import BasePredictor, Input
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from huggingface_hub import hf_hub_download
import json

class MultiTaskBERTModel(nn.Module):
    def __init__(self, model_name, task_configs):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.task_heads = nn.ModuleDict()
        self.task_configs = task_configs
        
        hidden_size = self.bert.config.hidden_size
        
        for task_name, config in task_configs.items():
            if config['type'] == 'classification':
                if config['num_classes'] == 2:
                    self.task_heads[task_name] = nn.Sequential(
                        nn.Linear(hidden_size, hidden_size // 2),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(hidden_size // 2, 1)
                    )
                else:
                    self.task_heads[task_name] = nn.Sequential(
                        nn.Linear(hidden_size, hidden_size // 2),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(hidden_size // 2, config['num_classes'])
                    )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        
        task_outputs = {}
        for task_name, head in self.task_heads.items():
            task_outputs[task_name] = head(pooled_output)
        
        return task_outputs

class Predictor(BasePredictor):
    def setup(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        
        # Download model from HuggingFace
        model_path = hf_hub_download(
            repo_id='alino-hcdc/calleval-bert',
            filename='best_calleval_bert_model.pth'
        )
        
        checkpoint = torch.load(model_path, map_location=self.device)
        task_configs = checkpoint.get('task_configs', self._get_default_task_configs())
        
        self.model = MultiTaskBERTModel('bert-base-uncased', task_configs)
        
        # Handle torch.compile prefixes
        state_dict = checkpoint['model_state_dict']
        if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
            new_state_dict = {}
            for key, value in state_dict.items():
                new_key = key[10:] if key.startswith('_orig_mod.') else key
                new_state_dict[new_key] = value
            state_dict = new_state_dict
        
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()
        
        self.task_configs = task_configs
    
    def _get_default_task_configs(self):
        return {
            'phase_classification': {'type': 'classification', 'num_classes': 3},
            'filler_detection': {'type': 'classification', 'num_classes': 2},
            'professional_greeting': {'type': 'classification', 'num_classes': 2},
            'verifies_patient_online': {'type': 'classification', 'num_classes': 2},
            'patient_verification': {'type': 'classification', 'num_classes': 2},
            'active_listening': {'type': 'classification', 'num_classes': 2},
            'asks_permission_hold': {'type': 'classification', 'num_classes': 2},
            'returns_properly_from_hold': {'type': 'classification', 'num_classes': 2},
            'shows_enthusiasm': {'type': 'classification', 'num_classes': 2},
            'sounds_polite_courteous': {'type': 'classification', 'num_classes': 2},
            'scheduled_appointment': {'type': 'classification', 'num_classes': 2},
            'recaps_time_date': {'type': 'classification', 'num_classes': 2},
            'offers_further_assistance': {'type': 'classification', 'num_classes': 2},
            'ended_call_properly': {'type': 'classification', 'num_classes': 2}
        }
    
    def predict(
        self,
        text: str = Input(description='Text to evaluate (call transcript or segment)')
    ) -> dict:
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=384,
            return_tensors='pt'
        )
        
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
        
        results = {}
        for task_name, output in outputs.items():
            config = self.task_configs[task_name]
            if config['type'] == 'classification':
                if config['num_classes'] == 2:
                    prob = torch.sigmoid(output).cpu().numpy()[0][0]
                    results[task_name] = {
                        'score': float(prob),
                        'prediction': 'positive' if prob >= 0.5 else 'negative'
                    }
                else:
                    probs = torch.softmax(output, dim=1).cpu().numpy()[0]
                    pred_class = int(torch.argmax(output, dim=1).cpu().numpy()[0])
                    results[task_name] = {
                        'predicted_class': pred_class,
                        'probabilities': probs.tolist()
                    }
        
        return results
