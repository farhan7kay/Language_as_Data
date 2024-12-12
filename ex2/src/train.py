import torch
import math
from src.helper import write_list_to_file

def train(model,num_epochs,optimizer,criterion,data_loader,path_to_save_folder,train_run_label,vocab_size,device,print_every=5):
    
    train_losses = []
    perplexities = []
    
    
    all_losses = []
    all_perp = []
    for epoch in range(num_epochs):
        total_loss = 0
        model.train()
        for batch_idx, (x, y) in enumerate(data_loader):
            x = x.to(device)
            y = y.to(device)
            
            optimizer.zero_grad()
            logits = model(x)
            
            loss = criterion(logits.view(-1, vocab_size), y.view(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % print_every == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx}/{len(data_loader)}], Loss: {loss.item():.4f}")
                all_losses.append(loss.item())
            if batch_idx % 50 == 0:
                torch.save(model.state_dict(), path_to_save_folder+"/temp_save")

                
                
        avg_loss = total_loss / len(data_loader)
        perplexity = math.exp(avg_loss)
        train_losses.append(avg_loss)
        perplexities.append(perplexity)
        print(f"Epoch [{epoch+1}/{num_epochs}] Average Loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f}")
        
        write_list_to_file(train_run_label+"_losses",train_losses,path_to_save_folder)
        write_list_to_file(train_run_label+"_step_losses",all_losses,path_to_save_folder)
        write_list_to_file(train_run_label+"_perplexities",perplexities,path_to_save_folder)
        torch.save(model.state_dict(), path_to_save_folder+"/"+train_run_label+"_model")
        return (all_losses,train_losses,perplexities)



def evaluate(model, dataloader,criterion,device,vocab_size): 
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits.view(-1, vocab_size), y.view(-1))
            total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    perplexity_simple = math.exp(avg_loss)
    return perplexity_simple
 