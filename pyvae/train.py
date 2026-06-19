from __future__ import annotations
import torch  
from torch.utils.data import DataLoader, TensorDataset  
from pyvae.models import InformedVAE

def train_ivae(  
    model: InformedVAE,
    x_train,
    x_val,
    epochs: int = 100,
    batch_size: int = 32,
    patience: int = 100,
    lr: float = 1e-5,
    device: str = "cpu",
) -> tuple[InformedVAE, dict]:

    model.to(device)

    x_train_tensor = torch.tensor(x_train.values, dtype=torch.float32)
    x_val_tensor = torch.tensor(x_val.values, dtype=torch.float32)

    train_dataset = TensorDataset(x_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    x_val_tensor = x_val_tensor.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-7)
    history = {"train": [], "val": []}
    best_val_loss = float("inf")     
    patience_counter = 0

    from tqdm.auto import tqdm 
    for epoch in tqdm(range(epochs)):
        model.train()                                
        epoch_train_loss = 0.0                      
        n_batches = 0

        for (x_batch,) in train_loader:            
            x_batch = x_batch.to(device)
        
            optimizer.zero_grad()                    
            recon, mu, log_var, h = model(x_batch)   
            loss = model.loss(x_batch, recon, mu, log_var, h)
            loss.backward()                          
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  
            optimizer.step()                         
        
            epoch_train_loss += loss.item()          
            n_batches += 1

        avg_train_loss = epoch_train_loss / n_batches  
    
        model.eval()                                
        with torch.no_grad():                 
            recon, mu, log_var, h = model(x_val_tensor)
            val_loss = model.loss(x_val_tensor, recon, mu, log_var, h).item()


        history["train"].append(avg_train_loss)
        history["val"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    return model, history



    

    
    
