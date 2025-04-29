# LLM evaluation and monitoring

## Introduction to AI Evaluation

When you build an AI system — whether it’s helping users write emails, answer questions, or generate code — you need to know: is it actually doing a good job? To figure that out, you need evaluations — or **"evals."** And not just once. Evaluations are essential throughout the entire AI system lifecycle.

- **During development**, you're experimenting: trying out different models, architectures, or prompts. Evals help make better decisions and design a system that performs well.  
- **In production**, it's about monitoring. You need to ensure the system keeps delivering high-quality results when real users rely on it.  
- **When making updates**, like changing a prompt or switching to a new model version, evaluations help you test for regressions and avoid making things worse.

This need for evaluation isn’t new. In fact, it’s a well-established practice in traditional machine learning — where you may be training models to solve tasks like spam detection or predicting sales volumes. In those cases, evaluation is typically objective: there's a correct answer, and you can directly measure whether the model got it right.

But things get more complex with LLM-powered applications. Instead of predicting a single outcome, these systems often help write text, summarize articles, translate languages, etc. And in these cases, there isn’t always one "correct" answer.  

Instead, you're often dealing with a range of possible responses — some better than others. Quality also often involves more subjective criteria, like **tone**, **clarity**, **helpfulness**. For example:

- If an AI generates code, how do you know it's not only correct, but also secure and efficient?  
- If an AI answers questions about your company's internal processes, how can you be sure it is accurate, up-to-date and is genuinely helpful to the user?

These kinds of tasks are what make generative systems so powerful — you can automate entirely new kinds of work. But this also adds significant complexity when it comes to evaluation.

> In this guide, we’ll explore how to evaluate AI systems built with large language models — from development to real-world monitoring.

## Chapter 1: Basics of AI system evaluation

In this chapter, we introduce two major types of AI tasks — **predictive (discriminative)** and **generative** — and explain how they shape the evaluation process. We also briefly cover how predictive systems are typically evaluated to learn from existing foundations.

### Discriminative vs Generative tasks 

AI systems and models can be broadly divided into two categories: **discriminative** and **generative**.  
Each serves a different purpose — and their evaluation methods reflect these differences.

### Discriminative tasks

Discriminative models focus on **making decisions or predictions**.

They take input data and classify it into predefined categories or predict numerical outcomes. In simpler terms, they’re great at answering **"yes or no," "which one," or "how many"** questions.

**Examples:**
- Classifying images as cat or dog.  
- Predicting the probability that an email is spam.  
- Identifying the sentiment of a movie review as positive or negative.  
- Predicting sales volumes.  

You can think of discriminative models as decision-makers. Their goal is to **predict the target variable** given a set of input features.

**Common categories of tasks include:**
- **Classification**: Predicting discrete labels (e.g., fraud vs. not fraud)  
- **Regression**: Predicting continuous values (e.g., monthly revenue)  
- **Ranking**: Predicting the relative ordering of items (e.g., search, recommendations)  

You would typically train narrow ML models on your own historical data to solve such predictive tasks. But you can also use pre-trained models — including LLMs — for some of these.  
For example, use an LLM to classify support queries into a set of predefined topics. In these cases, you can apply the same evaluation approaches as you would with traditional ML models.

### Generative tasks

Generative models, on the other hand, **create new data** that resembles the data they were trained on. Instead of classifying or predicting, they generate content — such as text, images, code, conversations, or even music.

**Examples:**
- Writing a creative story  
- Summarizing an article  
- Generating realistic images of non-existent people  
- Translating text from one language to another  
- Drafting a response to an email using the provided information  

Generative models are **creators**. They learn the underlying patterns in the data and use that to produce something new.

#### Quick Comparison: Discriminative vs. Generative Tasks

|                        | **Discriminative Tasks**       | **Generative Tasks**            |
|------------------------|-------------------------------|----------------------------------|
| **Goal**               | Make decisions                 | Create new content               |
| **Output**             | Labels, scores or predictions  | Text, code, images, or audio     |
| **Evaluation**         | Clear, objective metrics       | Complex, subjective metrics      |

Many modern applications of LLMs including AI agents and chatbots fall under **generative tasks**.

Before we dive into evaluating generative tasks, let’s first quickly review how this works for more traditional **predictive (discriminative)** systems. This helps with understanding core evaluation principles.

## Evaluating predictive systems

The process of evaluating discriminative systems is well-established and relatively straightforward.

### Evaluation process

**Offline evaluation**. Evaluation typically happens **offline**: you run an evaluation script or pipeline as you experiment and build your ML model. You assess its performance using a designated **test dataset**.

**Ground truth dataset**. You make the evaluation by comparing model predictions against expected labels or scores. For that, you need a dataset that represents your use case. It can be manually labeled or curated from historical data. 

In fact, such a dataset is a **prerequisite** for ML model development: you need it first to train the model itself! You then use part of this same dataset for evaluation. Typically, the data is split into:
- **Training set**: used to fit the model.  
- **Validation (or development) set**: used to tune hyperparameters and guide model improvements.  
- **Test set**: reserved for evaluating final model performance on unseen examples.  

For example, you might work with a collection of emails labeled as spam or not, or a historical record of product sales transactions. After training the model using most of the data, you evaluate it by comparing its predictions against the known outcomes in the remaining test set.

### Evaluation metrics

Depending on the type of task and the types of errors you prioritize, you can choose from several **standard evaluation metrics**. Here are examples based on the type of the task.

- **For regression tasks**, common evaluation metrics include:  
  - Mean Absolute Error (MAE)  
  - Mean Squared Error (MSE)  
  - Root Mean Squared Error (RMSE)  
  - R-Squared (Coefficient of Determination)  
  - Mean Absolute Percentage Error (MAPE)

- **For classification tasks**, typical metrics are:  
  - Accuracy  
  - Precision  
  - Recall  
  - F1 Score  
  - Logarithmic Loss  
  - AUC-ROC

- **For ranking tasks**, you often use metrics like:  
  - Mean Reciprocal Rank (MRR)  
  - Normalized Discounted Cumulative Gain (NDCG)  
  - Precision at K  

**Further reading:** You can find more detailed guides on [Classification Metrics](https://www.evidentlyai.com/classification-metrics) and [Ranking Metrics](https://www.evidentlyai.com/ranking-metrics).

While there is some diversity in metrics, the evaluation process for predictive systems is relatively straightforward: there is a single "right" answer and you can essentially quantify how many predictions were correct or close enough. (In tasks like multi-label classification, where multiple correct labels can apply to a single instance, similar metrics are adapted to evaluate across all relevant labels.)

### Production monitoring

In production settings, it’s often not possible to immediately verify whether model predictions are correct. Real labels are often only available after some time — either through manual review or once real-world outcomes are known.

You typically continue monitoring model quality by:

- **Collecting these delayed labels or actual values.**  You can manually label a sample of model outputs or collect real-world outcome data as it becomes available. For example, after predicting the estimated time of delivery (ETA), you can wait and capture the actual delivery time. This allows you to periodically recompute evaluation metrics, such as mean error, accuracy etc.

- **Using proxy metrics.** When labels are unavailable or too slow to collect at scale, you can rely on indirect monitoring signals, such as:
  - **Prediction drift monitoring**, which tracks changes in the distribution of model outputs over time (e.g., helps detect a shift toward more frequent fraud predictions).
  - **Input data quality checks**, which helps make sure that incoming data is valid and not corrupted.
  - **Data drift monitoring**, which tracks changes in input data distributions to verify that the model operates in a familiar environment.

**Further reading:** Learn more about [data drift](https://www.evidentlyai.com/ml-in-production/data-drift). 

### Evaluating LLM outputs

Whenever you use LLMs for predictive tasks, you can apply the same evaluation approaches as with traditional machine learning models — both offline and online. However, evaluating **generative tasks** is more complex. When multiple valid answers are possible, you can't simply check if an answer exactly matches a given reference. You need more complex comparative metrics as well as an option to evaluate more subjective quality dimensions such as clarity or helpfulness.

> In the next chapter, we’ll introduce methods and tools for evaluating generative tasks.

## Chapter 2: Evaluating generative systems

When building generative systems, the typical starting point is **manual evaluation**, where you manually review or label outputs. This step is essential: manual labeling helps you understand the types of errors your system produces and define your quality criteria. However, manual evaluation has limited scalability, especially when you run iterative experiments like trying different versions of prompts. **Automated evaluations** help speed up and scale the process. 

In this chapter, we will discuss approaches to automated evaluation of LLM system outputs, and types of metrics you can use.

### Reference-based vs Reference-free evaluations

There are two main evaluation workflows, depending on the stage and nature of your AI system:

- **Reference-based evaluation**  
- **Reference-free evaluation**

### Reference-based evaluation

This approach relies on having **ground truth answers**, and evaluation measures if the new outputs match the ground truth. Essentially, it follows the same principles as in traditional predictive system evaluation.

Reference-based evaluations are conducted **offline**:
- When you re-run evaluations after making changes, such as editing prompts during experiments.  
- Before pushing updates to production, as part of regression testing.

You start by preparing a custom evaluation dataset of expected inputs and outputs — for example, a set of questions you expect users to ask and their ideal responses. This dataset is often referred to as a **"golden set."** It should be representative of your real use cases. The quality of your evaluations will directly depend on how well it reflects the tasks your system must handle.

It is important to expand your golden set over time to keep it relevant as your product evolves or you discover new scenarios or edge cases. (But keep in mind that you cannot directly compare evaluation metrics across different golden sets if you change the underlying data you are testing on.)

Since multiple valid answers are often possible, you cannot rely solely on exact match metrics. Instead, specialized methods such as **semantic similarity scoring** or **LLM-as-a-judge** are used to assess the closeness or correctness of the model's outputs relative to the reference.

> We’ll cover specific approaches in the following sections.

### Reference-free evaluation

Reference-free methods directly assign quantitative scores or labels to the generated outputs without needing a ground truth answer.

This workf for both **offline and online testing**, when obtaining references isn’t possible or practical — for example:
- In complex, open-ended tasks like content generation  
- In multi-turn conversations  
- In production settings, where outputs are evaluated in real time  
- In certain evaluation scenarios like adversarial testing, where you assess the expected properties of an answer (e.g., by evaluating that it does not have any toxicity or bias)

Interestingly, LLMs that work with text data and generate open-ended outputs have more possibilities for reference-free evaluation compared to traditional ML models, which often deal with tabular data or non-interpretable features. With LLMs outputs, it is possible to assess **specific properties** of generated text — such as tone, fluency, or safety — even without having an exact ground truth reference.  This is enabled through methods like **LLM judges** and **predictive ML scoring models**.

**Read more**: [LLM Evaluation Guide](https://www.evidentlyai.com/llm-guide/llm-evaluation). Refer to this guide for additional explanations on different LLM evaluation workflows, such as comparative experiments, LLM stress-testing, red-teaming, and regression testing.

### Metrics and methods

Some LLM evaluation metrics — just like traditional predictive metrics — apply **only in reference-based scenarios**.  
Other methods, such as using **LLM judges**, can be used in both **reference-based** and **reference-free** evaluation.

Here are different LLM evaluation methods at a glance:

> 📖 **Source**: [LLM Evaluation Metrics Guide](https://www.evidentlyai.com/llm-guide/llm-evaluation-metrics).  
> You can refer to this guide for additional explanations on different LLM evaluation metrics.

In the following chapters, we will cover the following types of evaluation methods:

#### **Deterministic methods**
- **Text statistics**  
- **Pattern matching** with regular expressions  
- **Overlap-based metrics** like ROUGE and BLEU  

#### **Model-based methods**
- **Semantic similarity metrics** based on embedding models  
- **LLM-as-a-judge** approaches  
- **ML-based scoring models**  

### Dataset-level vs. Row-level evaluations

One more important distinction to make before we move into specific examples is between:

- Evaluations conducted at the dataset level
- Evaluations conducted at the individual input/output (row) level)

**Dataset-level** metrics aggregate results across all predictions and produce a single quality measure. This is typical for predictive tasks. In classic ML, we often use metrics like: Precision, Recall, F1 score. These metrics summarize performance across the full evaluation dataset — often with thousands or millions of examples.

**Row-level evaluators**, in contrast, focus on assessing each response individually. For example, LLM judges, ML models or semantic similarity evaluators provide a score or label per generated output — such as:
- Whether a response is correct or not  
- Sentiment score
- Similarity score

These numerical or categorical scores can be called **descriptors**. You can assign multiple descriptors to each input (or even a full conversation), evaluating aspects like relevance, tone, and safety at once.

**Score aggregation**. When working with row-level descriptors, you still need a way to combine individual scores into a performance summary across your test inputs.Sometimes it’s simple, such as:
- Averaging numerical scores  
- Counting the share of outputs that have a "good" label  

In other cases, you may need more complex aggregation logic. For instance:
- Set a threshold (e.g., flag any output with a semantic similarity score < 0.85 as "incorrect")  
- Calculate the share of correct responses based on that rule  

> When exploring evaluation methods below, we will focus primarily on **row-level evaluations**.  
> However, it is important to keep in mind your **aggregation strategy** as you run evals across multiple inputs in your dataset.
