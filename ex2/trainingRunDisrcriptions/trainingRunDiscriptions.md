```
from src.model import RegularizedLanguageModel
from src.trainComplete import TrainComplete
trainclass = TrainComplete(text_path = text_path,path_to_save_folder= path_to_save_folder,tokenizer = tokenizer,allowed_special=False)

model = RegularizedLanguageModel(vocab_size, embedding_dim, context_length, dropout=0.2).to(device)

trainclass.train(model,
              vocab_size,device,raw_text,"pers_standardLinear_ep4_batchsize16_evaluateevery20000",
                print_every=75,evaluate_every=20000,optimizer=None,criterion=None,
              batch_size = 16,
              embedding_dim = 128,
              context_length = 32,
              num_epochs = 4
             )
```


```
from src.model import RegularizedLanguageModel
from src.trainComplete import TrainComplete
trainclass = TrainComplete(text_path = text_path,path_to_save_folder= path_to_save_folder,tokenizer = tokenizer,allowed_special=False)

model = RegularizedLanguageModel(vocab_size, embedding_dim, context_length, dropout=0.2).to(device)

trainclass.train(model,
              vocab_size,device,raw_text,"pers_standardLinear_ep4_batchsize128_evaluateevery10000",
                print_every=75,evaluate_every=10000,optimizer=None,criterion=None,
              batch_size = 128,
              embedding_dim = 128,
              context_length = 32,
              num_epochs = 4
             )
```

```
# With Relu 
from src.model import LanguageModel

from src.model import RegularizedLanguageModel
from src.trainComplete import TrainComplete
trainclass = TrainComplete(text_path = text_path,path_to_save_folder= path_to_save_folder,tokenizer = tokenizer,allowed_special=False)

model = LanguageModel(vocab_size, embedding_dim, context_length, dropout=0.2).to(device)

trainclass.train(model,
              vocab_size,device,raw_text,"pers_LinearWithRelu_ep4_batchsize32_evaluateevery10000",
                print_every=75,evaluate_every=10000,optimizer=None,criterion=None,
              batch_size = 32,
              embedding_dim = 128,
              context_length = 32,
              num_epochs = 4
             )
```

```
from src.trainComplete import TrainComplete
from src.attentionModel import LanguageModelWithAttention

trainclass = TrainComplete(text_path = text_path,path_to_save_folder= path_to_save_folder,tokenizer = tokenizer,
                           allowed_special=False, is_attention_training = True)


context_length = 32  # Increased context size
embedding_dim = 128
attention_dim = 64
hidden_dim = 64
num_heads = 4

model = LanguageModelWithAttention(
    vocab_size, embedding_dim, attention_dim, context_length, hidden_dim, num_heads, dropout=0.2
).to(device)

trainclass.train(model,
              vocab_size,device,raw_text,"pers_attention_standard_dropout_ep10_eval10000",
                print_every=75,evaluate_every=10000,optimizer=None,criterion=None,
              batch_size = 32,
              embedding_dim = embedding_dim,
              context_length = context_length,
              num_epochs = 10
             )
```


```
from src.trainComplete import TrainComplete
from src.attentionModel import LanguageModelWithAttention

trainclass = TrainComplete(text_path = text_path,path_to_save_folder= path_to_save_folder,tokenizer = tokenizer,
                           allowed_special=False, is_attention_training = True)


context_length = 32  # Increased context size
embedding_dim = 128
attention_dim = 64
hidden_dim = 64
num_heads = 4

model = LanguageModelWithAttention(
    vocab_size, embedding_dim, attention_dim, context_length, hidden_dim, num_heads, dropout=0.2
).to(device)

trainclass.train(model,
              vocab_size,device,raw_text,"pers_attention_standard_dropout_ep10_eval10000",
                print_every=75,evaluate_every=10000,optimizer=None,criterion=None,
              batch_size = 32,
              embedding_dim = embedding_dim,
              context_length = context_length,
              num_epochs = 10
             )
 ```


```
from src.trainComplete import TrainComplete
from src.attentionModel import LanguageModelWithAttention

trainclass = TrainComplete(text_path = text_path,path_to_save_folder= path_to_save_folder,tokenizer = tokenizer,
                           allowed_special=False, is_attention_training = True)


context_length = 32  # Increased context size
embedding_dim = 128
attention_dim = 64
hidden_dim = 64
num_heads = 4

model = LanguageModelWithAttention(
    vocab_size, embedding_dim, attention_dim, context_length, hidden_dim, num_heads, dropout=0.2
).to(device)

trainclass.train(model,
              vocab_size,device,raw_text,"pers_attention_standard_dropout_ep10_eval10000",
                print_every=75,evaluate_every=10000,optimizer=None,criterion=None,
              batch_size = 32,
              embedding_dim = embedding_dim,
              context_length = context_length,
              num_epochs = 10
             )
```

```
from src.model import RegularizedLanguageModel
model = RegularizedLanguageModel(vocab_size, embedding_dim, context_length, dropout=0.2).to(device)

trainclass.train(model,
              vocab_size,device,raw_text,"standardLinear_ep4_batch32",
                print_every=75,evaluate_every=3000,optimizer=None,criterion=None,
              batch_size = 32,
              embedding_dim = 128,
              context_length = 32,
              num_epochs =  4
             )
```

```
# from src.model import RegularizedLanguageModel
model = RegularizedLanguageModel(vocab_size, embedding_dim, context_length, dropout=0.2).to(device)

trainclass.train(model,
              vocab_size,device,raw_text,"standardLinear_ep4_batch16",
                print_every=75,evaluate_every=3000,optimizer=None,criterion=None,
              batch_size = 16,
              embedding_dim = 128,
              context_length = 32,
              num_epochs =  4
             )
```


```
from src.model import RegularizedLanguageModel
model = RegularizedLanguageModel(vocab_size, embedding_dim, context_length, dropout=0.2).to(device)

trainclass.train(model,
              vocab_size,device,raw_text,"standardLinear_ep4_batch64",
                print_every=75,evaluate_every=3000,optimizer=None,criterion=None,
              batch_size = 64,
              embedding_dim = 128,
              context_length = 32,
              num_epochs =  4
             )
             
```    
             
             
             
```
from src.model import RegularizedLanguageModel
model = RegularizedLanguageModel(vocab_size, embedding_dim, context_length, dropout=0.2).to(device)

trainclass.train(model,
              vocab_size,device,raw_text,"standardLinear_ep4_batch8",
                print_every=75,evaluate_every=3000,optimizer=None,criterion=None,
              batch_size = 8,
              embedding_dim = 128,
              context_length = 32,
              num_epochs =  4
             )
```

```
from src.model import RegularizedLanguageModel
from src.helper import clean_text_spanish_both,clean_text_both,get_cleaned_spanish_text_as_string,clean_text_spanish_remove,get_lines_without_number,clean_spanish_text,get_cleaned_text

raw_text = get_cleaned_text(text_path,clean_text_spanish_remove)


model = RegularizedLanguageModel(vocab_size, embedding_dim, context_length, dropout=0.2).to(device)

trainclass.train(model,
              vocab_size,device,raw_text,"standardLinear_ep4_batch32_removeMethodPrep",
                print_every=75,evaluate_every=3000,optimizer=None,criterion=None,
              batch_size = 32,
              embedding_dim = 128,
              context_length = 32,
              num_epochs =  4
             )
```


```
from src.model import RegularizedLanguageModel
from src.helper import clean_text_spanish_both,clean_text_both,get_cleaned_spanish_text_as_string,clean_text_spanish_remove,get_lines_without_number,clean_spanish_text,get_cleaned_text

raw_text = get_cleaned_text(text_path,clean_text_spanish_both)


model = RegularizedLanguageModel(vocab_size, embedding_dim, context_length, dropout=0.2).to(device)

trainclass.train(model,
              vocab_size,device,raw_text,"standardLinear_ep4_batch32_removeMethodBoth",
                print_every=75,evaluate_every=3000,optimizer=None,criterion=None,
              batch_size = 32,
              embedding_dim = 128,
              context_length = 32,
              num_epochs =  4
             )
```

```
from src.trainComplete import TrainComplete
from src.attentionModel import LanguageModelWithAttention

trainclass = TrainComplete(text_path = text_path,path_to_save_folder= path_to_save_folder,tokenizer = tokenizer,
                           allowed_special=False, is_attention_training = True)


context_length = 32  # Increased context size
embedding_dim = 128
attention_dim = 64
hidden_dim = 64
num_heads = 4

model = LanguageModelWithAttention(
    vocab_size, embedding_dim, attention_dim, context_length, hidden_dim, num_heads, dropout=0.2
).to(device)

trainclass.train(model,
              vocab_size,device,raw_text,"pers_attention_standard_dropout_ep10_eval10000",
                print_every=75,evaluate_every=10000,optimizer=None,criterion=None,
              batch_size = 32,
              embedding_dim = embedding_dim,
              context_length = context_length,
              num_epochs = 10
             )
```


```
from src.model import RegularizedLanguageModel
from src.trainComplete import TrainComplete
from src.helper  import clean_pers_text_replace, get_cleaned_text,clean_pers_remove,clean_text_pers_both

raw_text = get_cleaned_text(text_path,clean_pers_remove)
trainclass = TrainComplete(text_path = text_path,path_to_save_folder= path_to_save_folder,tokenizer = tokenizer,allowed_special=False)

model = RegularizedLanguageModel(vocab_size, embedding_dim, context_length, dropout=0.2).to(device)


trainclass.train(model,
              vocab_size,device,raw_text,"pers_standardLinearNotRelu_ep4_evaluate10000_preprocessingRemove",
                print_every=75,evaluate_every=10000,optimizer=None,criterion=None,
              batch_size = 32,
              embedding_dim = 128,
              context_length = 32,
              num_epochs = 4
             )
```

```

from src.model import RegularizedLanguageModel
from src.trainComplete import TrainComplete
from src.helper  import clean_pers_text_replace, get_cleaned_text,clean_pers_remove,clean_text_pers_both

raw_text = get_cleaned_text(text_path,clean_text_pers_both)
trainclass = TrainComplete(text_path = text_path,path_to_save_folder= path_to_save_folder,tokenizer = tokenizer,allowed_special=False)

model = RegularizedLanguageModel(vocab_size, embedding_dim, context_length, dropout=0.2).to(device)


trainclass.train(model,
              vocab_size,device,raw_text,"pers_standardLinearNotRelu_ep4_evaluate10000_preprocessingBoth",
                print_every=75,evaluate_every=10000,optimizer=None,criterion=None,
              batch_size = 32,
              embedding_dim = 128,
              context_length = 32,
              num_epochs = 4
             )


raw_text = get_cleaned_text(text_path,clean_pers_text_replace)
```


```
from src.trainComplete import TrainComplete
from src.attentionModel import LanguageModelWithAttention

trainclass = TrainComplete(text_path = text_path,path_to_save_folder= path_to_save_folder,tokenizer = tokenizer,
                           allowed_special=False, is_attention_training = True)


context_length = 32  # Increased context size
embedding_dim = 128
attention_dim = 64
hidden_dim = 64
num_heads = 4

model = LanguageModelWithAttention(
    vocab_size, embedding_dim, attention_dim, context_length, hidden_dim, num_heads, dropout=0.2
).to(device)

trainclass.train(model,
              vocab_size,device,raw_text,"pers_attention_standard_dropout_batchsize16_ep5_eval10000",
                print_every=75,evaluate_every=10000,optimizer=None,criterion=None,
              batch_size = 16,
              embedding_dim = embedding_dim,
              context_length = context_length,
              num_epochs = 5
             )  
```


```
from src.trainComplete import TrainComplete
from src.attentionModel import LanguageModelWithAttention

trainclass = TrainComplete(text_path = text_path,path_to_save_folder= path_to_save_folder,tokenizer = tokenizer,
                           allowed_special=False, is_attention_training = True)


context_length = 32  # Increased context size
embedding_dim = 128
attention_dim = 64
hidden_dim = 64
num_heads = 4

model = LanguageModelWithAttention(
    vocab_size, embedding_dim, attention_dim, context_length, hidden_dim, num_heads, dropout=0.2
).to(device)

trainclass.train(model,
              vocab_size,device,raw_text,"pers_attention_standard_dropout_batchsize64_ep5_eval10000",
                print_every=75,evaluate_every=10000,optimizer=None,criterion=None,
              batch_size = 64,
              embedding_dim = embedding_dim,
              context_length = context_length,
              num_epochs = 5
             )
```


```
from src.trainComplete import TrainComplete
from src.attentionModel import LanguageModelWithAttention

trainclass = TrainComplete(text_path = text_path,path_to_save_folder= path_to_save_folder,tokenizer = tokenizer,
                           allowed_special=False, is_attention_training = True)


context_length = 32  # Increased context size
embedding_dim = 128
attention_dim = 64
hidden_dim = 64
num_heads = 4

model = LanguageModelWithAttention(
    vocab_size, embedding_dim, attention_dim, context_length, hidden_dim, num_heads, dropout=0.2
).to(device)

trainclass.train(model,
              vocab_size,device,raw_text,"pers_attention_standard_dropout_ep1_eval10000",
                print_every=75,evaluate_every=10000,optimizer=None,criterion=None,
              batch_size = 32,
              embedding_dim = embedding_dim,
              context_length = context_length,
              num_epochs = 1
             )
```


