# Student Help Desk: Long-Form Project Write-Up

This is my own write-up of the Student Help Desk project, a RAG-based AI assistant for answering student queries. I wrote it to explain the idea, the way the pipeline works, and what I learned.

## The problem

Students have lots of everyday questions. They want to know how a process works, where to find some information, or what a rule says. The answers usually exist somewhere in documents, but nobody enjoys searching through long files. A normal chatbot built only on a language model can sound confident while giving wrong answers, because it answers from what it learned during training and not from the real documents of the place the student belongs to. In AI this problem is called hallucination.

## The idea: Retrieval-Augmented Generation

Retrieval-Augmented Generation, or RAG, fixes this by splitting the work into two jobs. First, a retrieval step looks through a custom knowledge base and finds the passages that are most relevant to the question. Second, a generation step gives those passages to the language model and asks it to write the answer using them. The model becomes more like a student taking an open-book exam than someone guessing from memory.

## How the pipeline works

1. **Collect the knowledge base.** I gathered the information students would need into a custom knowledge base made of text documents.
2. **Chunk the documents.** Long documents are cut into smaller pieces so that each piece covers one idea. Small chunks are easier to match with a question and fit easily inside a prompt.
3. **Turn chunks into vectors.** Each chunk is converted into a list of numbers, called an embedding, that captures its meaning. Chunks with similar meanings end up close to each other.
4. **Retrieve.** When a student asks a question, the question is converted into a vector too, and the system finds the chunks that are closest in meaning. This is the vector-based retrieval part of the project.
5. **Generate.** The best chunks and the question are sent to the LLM with instructions to answer from the given context. The answer comes back in natural language.

## Why I used vector-based retrieval

Students never phrase a question the same way the document is written. Someone might ask "how do I get my marks rechecked" while the document says "re-evaluation request". A pure keyword search could miss this, because the words do not match. Vector retrieval looks at meaning, so it can connect the two sentences even without shared words. This is the main reason a vector-based retrieval pipeline improved the relevance of responses in my project.

## Reducing hallucinations

The assistant is designed to rely on the retrieved context. The more clearly the prompt tells the model to answer only from the context, and to admit when the context does not contain the answer, the fewer made-up answers appear. Good chunking and good retrieval matter just as much as the prompt, because if the wrong passage is retrieved, even a well-behaved model will give a weak answer.

## What I learned

- Retrieval quality decides answer quality. If the right chunk is not found, the LLM cannot answer correctly.
- Chunk size matters. Chunks that are too big mix many topics together, and chunks that are too small lose context.
- Prompting is part of the engineering. Clear instructions such as "use only the context" make a visible difference.
- Testing with real questions is the best way to find weak spots. I learned to try questions phrased in different ways, not just the obvious ones.
- Documentation matters. Writing down how the pipeline works helped me understand it better and made it easier to explain to others.

## What I would improve

If I continued working on Student Help Desk, I would combine keyword search with vector search, which is called hybrid retrieval, because exact terms like names and codes are sometimes better matched by keywords. I would also add a way to show the source of each answer, so students can check the original document themselves. This digital clone project actually uses exactly that hybrid idea.
