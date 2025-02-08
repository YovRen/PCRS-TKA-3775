# PCRS-TKA

This is the official PyTorch implementation for the paper:

> Enhancing Conversational Recommender Systems with Tree-Structured Knowledge and Pretrained Language Models

## Requirements

- accelerate==0.29.3
- nltk==3.8.1
- numpy==2.1.1
- torch==2.2.1+cu118
- torch_geometric==2.6.1
- tqdm==4.66.2
- transformers==4.38.2
- swanlab==0.3.21

## Download Model

Please download RoBERTa-base from the [link](https://huggingface.co/FacebookAI/roberta-base), move it into `model/roberta-base`.

Please download DialoGPT-small from the [link](https://huggingface.co/microsoft/DialoGPT-small), move it into `model/dialogpt-small`.

## Process Data

Please download DBpedia from the [link](https://databus.dbpedia.org/dbpedia/mappings/mappingbased-objects/2021.09.01/mappingbased-objects_lang=en.ttl.bz2), after unzipping, move it into `data/dbpedia`.

```python
cd data
python dbpedia/extract_kg.py

# inspired
python inspired/extract_subkg.py
python inspired/process_dataset.py

# redial
python redial/extract_subkg.py
python redial/process_dataset.py
```

## Quick-Start

We run all experiments and tune hyperparameters on a GPU with 24GB memory, you can adjust `per_device_train_batch_size` and `per_device_eval_batch_size` according to your GPU, and then the optimization hyperparameters (e.g., `learning_rate`) may also need to be tuned.

### Recommendation Subtask

```bash
cd src
python
    --dataset inspired \  # [inspired, redial]
    --rec_pre_num_train_epoch 5 \
    --rec_pre_learning_rate 6e-4 \ # 5e-4 for redial
    --rec_pre_batch_size 64 \
    --rec_num_train_epoch 5 \
    --rec_learning_rate 1e-4 \
    --rec_batch_size 64 \
```

### Conversation Subtask

```bash
cd src
python
    --dataset inspired \  # [redial, inspired]
    --conv_pre_num_train_epoch 3 \
    --conv_pre_learning_rate 6e-4 \ # 5e-4 for redial
    --conv_pre_batch_size 8 \
    --conv_num_train_epoch 2 \
    --conv_learning_rate 1e-4 \
    --conv_batch_size 8 \
```

### Human Evaluation

We implemented multiple models (ReDial, KGSF,UniCRS, PCRS-TKA) and designed a questionnaire consisting of 100 questions randomly selected from the datasets. Each question was paired with one real response and four model-generated responses(anonymous). We invited ten annotators to manually evaluate the responses based on three criteria, with scores ranging from 0 to 5:

1. **Fluency** (Can the system engage in smooth and uninterrupted communication?)
2. **Informativeness** (Is the system's recommendation helpful for you?)
3. **Consistency** (Whether the system answers appropriately or responds off-topic, compared to the real user answer)
4. **Accuracy** (If the information mentioned in the response correct or matched with the knowledge graph)

The questionnaire and the users' original rating files are stored in the `others/survey` directory. Below is an example that can be manually evaluated based on Fluency, Question-Answer Consistency, and Informativeness:

```
Context: 
    System: Hello, how are you doing today?
    User: Good
    System: Great, glad to hear that, so do you like to watch movies? what type of movies do you like?
    User: I like mostly comedy movies
    System: Do you like comedy with action movies?
    User: I like those kind of movies sometimes
    System: I see so mostly comedy movies, I like those as well
    User: That's very nice. What titles you like?
    System: I like comedy movies as well but my favorite genre is action and also superhero movies, I really like those.
    User: Like the Marvel movies?
    System: Yeah like Marvel movies, exactly, did you happen to watch Deadpool? that movie is great and with a lot of comedy in it.
    User: I only saw some scenes of it.
    System: Then you will really like it, Ryan Reynolds is the leading actor and he is really funny. There are 2 parts so far and the third one is coming.
    User: I'll check those whenever whenever I get the chance.
Real Response: Yeah you definitely should, he is my favorite hero/anti hero
```

> | **Response**                                                                                                                                               | **Fluency** | Informativeness | Consistency | Accuracy |
> | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------- | --------------- | ----------- | -------- |
> | I have n't seen it .                                                                                                                                             | 3                 | 2               | 2           | 3        |
> | I would be a movie ! I will be the trailer .                                                                                                                     | 2                 | 2               | 2           | 3        |
> | So would you like me to recommend a movie for you?                                                                                                               | 3                 | 2               | 2           | 3        |
> | You should check out the movie with Ryan Reynolds and see if you like it. It is a really funny movie with a good plot and a lot action and comedy type of story. | 4                 | 3               | 3           | 4        |

### Baseline Configuration

The baseline models are implemented using the [CRSLab tool](https://github.com/RUCAIBox/CRSLab). And our configuration files are stored in `others/crslab-config`.
