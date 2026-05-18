import os
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from model import MolecularGAT

class EnsembleTester:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = []
        base_dir = "./save_models"
        subsets = ["subset_01","subset_02","subset_03","subset_04","subset_05",
                   "subset_06","subset_07","subset_08","subset_09","subset_10"]
        for subset in subsets:
            model_path = os.path.join(base_dir, subset, "final.pt")
            model = MolecularGAT(num_layers=6)  
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.eval()
            model = model.to(self.device)
            self.models.append(model.to(self.device))
    
    def run_test(self, subset_name):
        test_data = torch.load(f"./data/{subset_name}/{subset_name}.pt", weights_only=False)
        loader = DataLoader(test_data, batch_size=16)
        
        results = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                probs = torch.zeros((len(batch.ID), 1), device='cpu')               
                for model in self.models:
                    outputs = model(batch)
                    probs += torch.sigmoid(outputs).cpu()
                probs /= len(self.models)

                ID_list = [str(ID) for ID in batch.ID]
                for i in range(len(batch.ID)):
                    results.append({
                        "ID": ID_list[i],
                        "pred_prob": probs[i].item()
                    })
        
        save_dir = "./result"
        os.makedirs(save_dir, exist_ok=True)
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(save_dir, f"{subset_name}.csv"), index=False)
        print(f"{subset_name} testing completed")

if __name__ == "__main__":
    subsets = ["QSIDB", "DrugBank", "MiMeDB"]
    tester = EnsembleTester()
    for subset in subsets:
        tester.run_test(subset)