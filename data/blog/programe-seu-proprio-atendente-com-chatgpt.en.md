---
title: Write your own chatbot for Costumer Service based on ChatGPT
date: '2023-09-15'
tags: ['chatgpt', 'llm', 'customer-service']
draft: false
summary: Details on how to build a chatbot for costumer service based on ChatGPT
authors: ['cleto']
---

![Untitled](/static/images/chatgpt.webp)

Being the first blog post from GridoAI, I want to introduce you to one of the countless applications of LLMs (Large Language Models): Customer service. Automated customer service is particularly valuable for companies facing a large volume of inquiries, as it allows managing multiple requests simultaneously, offering quick and consistent responses at any time of the day.

ChatGPT itself is a fascinating product, but it cannot serve your customers without knowing about your company or your product/service. That's where we come in. How do you inform an LLM about the information it needs to perform its function? There are two main ways. This will be a serious dev-to-dev talk!

### Before we continue

- Token: You can think of tokens as pieces of words, where 1,000 tokens are roughly equivalent to about 750 words. It's the basic unit of text for LLMs.
- Context: Context is how far an LLM can complete texts. If an LLM has 100 tokens of context and you provide a text with 30 tokens, it can complete a maximum of 70 tokens.

## **Adding instructions in the context**

### Tokens and Context

The main model behind ChatGPT is gpt-3.5-turbo. It has two versions, one with 4,000 tokens of context, which is approximately 3,000 words, and one with 16,000 tokens, which is about 12,000 words. Depending on the complexity of your service, it's possible to instruct the LLM within these limits. For example, if your intention is to answer questions based on an FAQ (Frequently Asked Questions) and it contains up to 10,000 words, you can create a virtual attendant based on that FAQ!

### Code Example

Based on an FAQ I found on the [internet](https://www.gov.br/empresas-e-negocios/pt-br/observatorioapl/faq), here's a code example:

```python
import openai

openai.api_key = "YOUR_API_KEY"

faq = """
Question: Is the APL a closed group of companies?
Answer: No. It is open to all companies whose main activity is similar
and is present in the geographical area of the APL. The APL behaves
like an open system, where relationships with the external environment promote its
continuous strengthening, and the sum of the parts represents
more than the whole. The interaction between the parts, and these with the external environment,
constitutes multiple relationships that expand the growth possibilities
of this system.

Question: What is a Support Institution?
Answer: Every institution that supports the activities of articulation,
interaction, development, and integration. When talking about supporting the
development of a certain sector, it is necessary to bring together different
institutions that can contribute to the support and strengthening of that
sector. In this context, there are business associations, public power,
universities, technological centers, workforce training schools, financial agents, development agencies,
among others.
...
"""

system_message = f"""
You are a representative of the Brazilian government about Observatório APL.
Your goal is user satisfaction.
Here's some content to assist you:
{faq}
"""

first_message = {"role": "system", "content": system_message}

def answer(question):
    messages = [first_message, {"role": "user", "content": question}]
    chat = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)
    reply = chat.choices[0].message.content
    return reply

```

In production, the implementation wouldn't be this way. For two reasons:

- You need to develop an interface for your customer to interact with your bot, which is usually a front-end on a web page. Therefore, the `answer` function should be connected to a web framework like Django, Flask, or FastAPI.
- If your service is indeed a conversation, you need to provide the entire conversation history in all API calls. This means that your function should receive `messages` instead of `question`, where the question would be the last message in the list. This gives the LLM the context of the conversation and allows for contextualized answers.

Note that the `system_message` variable has a prompt that guides the LLM on how to serve your customer. You can customize this prompt in a way that best suits your use case.

## **Search in documents**

### Introduction to Semantic Search

What if your service instructions don't fit within 4,000 tokens or even 16,000 tokens of context? In that case, you need to select parts of your service documents depending on the customer's question. This seems much more complex, right? Here's a diagram of how this should work in practice.

![Untitled](/static/images/rag-ilustration.png)

> Vector? Vector database? What are these things?

These are the exact elements needed to perform a **semantic search**. I'll go into details shortly, but first, some context.

Semantic search is a search strategy that considers the meaning of the provided text. This means that "I like cats" and "Love felines" will be considered similar even with completely different words! In practice, this means that if the customer's question is not entirely contained in words in a question and answer document, but the meaning is close, this document will be selected to contextualize the question. Magnificent, isn't it?

### Vector

A vector is a "list of numbers", and the _dimension_ of the vector is the number of elements in that list. Therefore, `(1, 3)` is a vector with dimension 2. Two-dimensional vectors can be represented on a plane like this:

![Untitled](/static/images/plano2d.png)

Note that the vector `(1, 3)` is much further from the vector `(5, -2)` than from `(-1, 2)`. This concept of **distance** is extremely relevant, but there are several mathematical ways to interpret it. Here we will only consider cosine distance.

### Cosine Distance

Also known as cosine dissimilarity. The concept is simple. The larger the angle between two vectors, the more distant they are considered. When considering this interpretation of distance, the minimum distance is 0 (when the angle between them is 0°) and the maximum distance is 2 (when the angle between them is 180°).

![Untitled](/static/images/distancia-cosseno.png)

Now imagine if you could represent texts (questions and answers) as vectors. Of course, just two dimensions wouldn't be enough to represent all the complexity that texts can have. But even if you use more dimensions, the concept of _distance_ still applies. This way, it would be possible to calculate the segments of documents that are less _distant_ from the customer's question and use them as context, and not the rest.

### Embeddings

The good news is that there **is** a way to represent texts as vectors, and it's called **embeddings**. There are several models for calculating embeddings available on the internet.

OpenAI offers the `text-embedding-ada-002` and charges $0.1 per million tokens. Remember, these are approximately 750,000 words, which in turn are approximately 3,000 book pages, and you would need to calculate embeddings of the complete service material and also every question sent to your chatbot.

> Ok, I understand how the search works, but where do I store these vectors?

### Vector Database

They specialize in this task and recently even started calling themselves "long-term memory for AIs". There are several options, each with its characteristics. Here's a list of some options if you want to evaluate:

- Pinecone
- Milvus
- Qdrant
- pgvector (It's an extension of Postgres that adds vectors)

I won't focus on the pros and cons of each, as that's not the main point.

### Code Example

Given this avalanche of information, let's get our hands dirty! First, we'll need a function that adds new texts to the "AI's memory". This is the `add_text_to_db`.

```python
def add_to_db(text, vec):
		# The implementation depends on the database used
    ...

def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\\n", " ")
    return openai.Embedding.create(input=[text], model=model)["data"][0]["embedding"]

def add_text_to_db(text):
    vec = get_embedding(text)
    add_text_to_db(text, vec)

```

Another disclaimer regarding the implementation in production: pay attention to adding texts of similar size to the size of the query that will be made. Calculating the embedding of entire pages may not be as useful if your questions are up to 5 words, for example. If you have a very large text and you can't segment it manually, consider another strategy called _chunking_. I won't address it here to keep it short, but it's also very interesting!

Now we can redo our `answer` function.

```python
def get_near_docs(vec):
    # The implementation depends on the database used
    ...

def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\\n", " ")
    return openai.Embedding.create(input=[text], model=model)["data"][0]["embedding"]

def answer(question):
    vec = get_embedding(question)
    related_docs = get_near_docs(vec)
    docs_str = "\\n".join(related_docs)
    messages = [
        {
            "role": "system",
            "content": f"""
                You are a representative of company X. Your goal is customer satisfaction.
                Here's some content to assist you:
                {docs_str}
            """,
        },
        {"role": "user", "content": question},
    ]
    chat = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)
    reply = chat.choices[0].message.content
    return reply

```

## Conclusion

Unfortunately, the second approach is not as simple as the first, but it's much more powerful. Therefore, I encourage you to try implementing it. It's very valuable both for your learning and for your company, which will save a good amount of money on customer service.

### Next Steps

Too hard? Don't despair! Even though it seems quite complex and this is just the tip of the iceberg when talking about intelligent chatbots, this mechanism and many other sophistications are already publicly available and **free** in our intelligent chatbot, [Grido](http://gridoai.com/).

Note: One dimension of complexity not addressed here is how to integrate with existing and mature costumer service platforms. We propose to integrate with whatever system you use, no matter what it is!