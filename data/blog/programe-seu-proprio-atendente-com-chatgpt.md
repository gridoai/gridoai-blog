---
title: Programe seu próprio atendente com o ChatGPT
date: '2023-09-15'
tags: ['chatgpt', 'llm', 'costumer-service']
draft: false
summary: Detalhes sobre como construir um atendente virtual baseado no ChatGPT
authors: ['cleto']
---

![Untitled](/static/images/chatgpt.webp)

Sendo o primeiro post do blog da GridoAI, quero te introduzir numa das infinitas aplicações dos LLMs (Large Language Models): Atendimento ao cliente (ou lead). O atendimento ao cliente automatizado é particularmente valioso para empresas que enfrentam um grande volume de consultas, pois permite gerenciar múltiplos pedidos simultaneamente, oferecendo respostas rápidas e consistentes a qualquer hora do dia.

O ChatGPT por si só já é um produto fascinante, mas não tem como ele atender seus clientes sem saber do que se trata sua empresa ou seu produto/serviço e é aí que a gente entra. Como informar um LLM sobre as informações que ele necessita para efetuar sua função? Existem duas principais formas. Isso vai ser um papo sério de dev para dev!

### Antes de continuar

- Token: Você pode pensar nos tokens como pedaços de palavras, onde 1.000 tokens equivalem a cerca de 750 palavras. É a unidade básica de texto dos LLMs.
- Contexto: Contexto é até onde um LLM consegue completar os textos. Se um LLM tem 100 tokens de contexto e você fornece um texto com 30 tokens, ele vai conseguir completar no máximo 70 tokens.

## **Adicionando instruções no contexto**

### Tokens e Contexto

O principal modelo por de trás do ChatGPT é o gpt-3.5-turbo. Este tem duas versões, a de 4.000 tokens de contexto, que é aproximadamente 3.000 palavras e a de 16.000 tokens que é aproximadamente 12.000 palavras. Dependendo da complexidade do seu atendimento, é possível instruir o LLM dentro de algum desses limites. Por exemplo, se a sua intenção é responder perguntas baseadas num FAQ (Perguntas frequentes) e este conter por volta de 10.000 palavras no máximo, é possível criar um atendente virtual com base nesse FAQ!

### Exemplo de código

Baseado num FAQ que encontrei na [internet](https://www.gov.br/empresas-e-negocios/pt-br/observatorioapl/faq), segue exemplo de código:

```python
import openai

openai.api_key = "YOUR_API_KEY"

faq = """
Pergunta: O APL é um grupo fechado de empresas?
Resposta: Não. Está aberto a todas as empresas cuja atividade fim seja semelhante
e tenha presença na área de atuação geográfica do APL. O APL comporta-se
como um sistema aberto, onde as relações com o meio externo promovem o
fortalecimento contínuo do mesmo, sendo que, a soma das partes representa
mais que o todo. A interação entre as partes, e estas com o meio externo,
constitui relações múltiplas que expandem as possibilidades de crescimento
desse sistema.

Pergunta: O que é uma Instituição de Apoio?
Resposta: Toda instituição que tem como função o apoio às atividades de articulação,
interação, desenvolvimento e integração. Quando se fala em apoiar o
desenvolvimento de um determinado setor, é necessário congregar diferentes
instituições que possam contribuir para o apoio e o fortalecimento desse
setor. Nesse contexto, estão as associações empresariais, o poder público,
as universidades, os centros tecnológicos, as escolas de formação de
mão-de-obra, os agentes financeiros, os órgãos de desenvolvimento,
entre outras.
...
"""

system_message = f"""
Você é um atendente do governo brasileiro sobre Observatório APL.
Seu objetivo é a satisfação do usuário.
Segue conteúdo para auxílio:
{faq}
"""

first_message = {"role": "system", "content": system_message}

def answer(question):
    messages = [first_message, {"role": "user", "content": question}]
    chat = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)
    reply = chat.choices[0].message.content
    return reply

```

Em produção, a implementação não seria essa. Por dois motivos:

- É necessário desenvolver uma interface para o seu cliente interagir com seu bot, esta que normalmente é um front-end numa página web. Dado isso, a função `answer` deveria estar conectada um _web framework_ com Django, Flask ou FastAPI.
- No caso do seu atendimento ser de fato uma conversa, é necessário ser fornecido todo o histórico da conversa em todas as chamadas de API. Isso significa que sua função deve receber `messages` ao invés de `question` onde a pergunta seria a última mensagem da lista. Isso dá o contexto da conversa para o LLM e permite respostas contextualizadas.

Note que a variável `system_message` tem um _prompt_ que orienta o LLM sobre como atender o seu cliente. É possível personalizar este _prompt_ da forma que for melhor para o seu caso de uso.

## **Busca em documentos**

### Introdução à Busca Semântica

E se as suas orientações de atendimento não cabem em 4.000 tokens e nem em 16.000 tokens de contexto? Nesse caso, você precisa selecionar parte dos seus documentos de atendimento a depender da pergunta do cliente. Isso parece bem mais complexo, não é mesmo? Segue diagrama de como isso deve funcionar na prática.

![Untitled](/static/images/ilustracao-rag.png)

> Vetor? Banco de vetores? O que são essas coisas?

Esses são os exatos elementos necessários para efetuar uma **busca semântica**. Já vou entrar em detalhes, mas antes vou te contextualizar.

Busca semântica é uma estratégia de busca que leva em consideração o significado do texto fornecido. Isso significa que “Eu gosto de gatos” e “Amo felinos” serão considerados parecidos mesmo com palavras completamente diferentes! Na prática, isso significa que se a pergunta do cliente não estiver inteiramente contida em palavras num documento de perguntas e respostas, mas o sentido for próximo, este documento será selecionado para contextualizar a pergunta. Magnífico, não é?

### Vetor

Vetor é uma “lista de números” e a _dimensão_ do vetor é a quantidade de elementos nessa lista. Portanto, `(1, 3)` é um vetor com dimensão 2. Vetores de dimensão 2 podem ser representados num plano como esse:

![Untitled](/static/images/plano2d.png)

Note que o vetor `(1, 3)` está muito mais distante do vetor `(5, -2)` do que do `(-1, 2)`. Esse conceito de **distância** é extremamente relevante, mas existem diversas formas matemáticas de interpretá-lo. Aqui iremos considerar apenas a distância por cossenos.

### Distância por Cosseno

Também conhecido por dissimilaridade de cossenos. O conceito é simples. Quanto maior o ângulo entre dois vetores, mais distantes eles são considerados. Ao considerar essa interpretação de distância, a distância mínima é 0 (quando o ângulo entre eles é 0°) e a distância máxima é 2 (quando o ângulo entre eles é 180°).

![Untitled](/static/images/distancia-cosseno.png)

Agora imagine se você pudesse representar textos (perguntas e respostas) como vetores. Claro que apenas duas dimensões não seriam suficientes para representar todo a complexidade que os textos podem ter. Mas mesmo que você use mais dimensões, o conceito de _distância_ ainda vale. Dessa forma, seria possível calcular os segmentos de documentos que são menos _distantes_ da pergunta do cliente e utilizá-los como contexto, e não o resto.

### Embeddings

A boa notícia é que **existe** uma forma de representar textos como vetores e se chama **embeddings**. Existem diversos modelos para cálculo de embeddings disponíveis na internet.

A OpenAI disponibiliza o `text-embedding-ada-002` e custa $0.1 por milhão de tokens. Lembre-se que isso são aproximadamente 750 mil palavras que por sua vez são aproximadamente 3.000 páginas de livro e que seria necessário calcular embeddings do material de atendimento por completo e também de toda pergunta que enviarem ao seu chatbot.

> Ok, já entendi como a busca funciona, mas onde eu armazeno esses vetores?

### Banco de vetores

Eles são especializados nessa tarefa e recentemente estão até se autodenominando “memória de longo prazo para IAs”. Existem diversas opções e cada um com suas características. Segue lista de algumas opções caso queira avaliar:

- Pinecone
- Milvus
- Qdrant
- pgvector (É uma extensão do Postgres que adiciona vetores)

Não vou me atentar aos prós e contras de cada um, pois esse não é o foco.

### Exemplo de código

Dado essa avalanche de informações, vamos botar a mão na massa! Primeiro vamos precisar de uma função que adiciona novos textos na “memória da IA”. Essa é a `add_text_to_db`.

```python
def add_to_db(text, vec):
		# A implementação depende do banco utilizado
    ...

def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\\n", " ")
    return openai.Embedding.create(input=[text], model=model)["data"][0]["embedding"]

def add_text_to_db(text):
    vec = get_embedding(text)
    add_text_to_db(text, vec)

```

Outro disclaimer com relação à implementação disso em produção: se atente a adicionar textos de tamanho parecido com o tamanho da consulta que será feita. Calcular o embedding de páginas inteiras pode não ser tão útil se suas perguntas são de até 5 palavras, por exemplo. Se você tem um texto muito grande e você não pode segmentar ele manualmente, considere outra estratégia chamada _chunking_. Não vou abordar aqui para não me estender muito, mas também é bastante interessante!

Agora podemos refazer nossa função `answer`.

```python
def get_near_docs(vec):
    # A implementação depende do banco utilizado
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
                Você é um atendente da empresa X. Seu objetivo é a satisfação do cliente.
                Segue conteúdo para auxílio:
                {docs_str}
            """,
        },
        {"role": "user", "content": question},
    ]
    chat = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)
    reply = chat.choices[0].message.content
    return reply

```

## Conclusão

Infelizmente a segunda abordagem não é tão simples quanto a primeira, mas é muito mais poderosa. Por isso, te encorajo a tentar implementar. É bastante valioso tanto para seu aprendizado quanto para sua empresa, que vai economizar uma boa grana com atendimento.

### Próximos passos

Muito difícil? Não se desespere! Mesmo parecendo bastante complexo e essa ser só a ponta do iceberg quando se fala de chatbots inteligentes, esse mecanismo e muitas outras sofisticações já estão disponíveis publicamente e de forma **gratuita** no nosso chatbot inteligente, a [Grido](http://gridoai.com/).

Obs: Uma dimensão de complexidade não abordada aqui é com relação a como integrar com plataformas de atendimento ao cliente já existentes e maduras. A gente se propõem a integrar com o sistema que você usa seja ele qual for!
