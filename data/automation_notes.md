# My Study Notes: Automation, APIs and n8n

These are notes I wrote for myself while learning automation and integration. I wrote them in simple words so they are easy to revise later.

## What is workflow automation?

Workflow automation means letting software do repetitive tasks by itself. Instead of copying data from one app to another by hand, you build a workflow: a chain of steps where each step takes input, does something and passes the result on. A workflow has a trigger (what starts it), some actions in the middle, and an output at the end.

## n8n

n8n is the workflow automation tool I use most. It is a visual, node-based tool. Each node is one step, such as receiving data, changing it, calling another service or sending a message. You connect nodes with lines, and the data flows from left to right. It is a great example of low-code work, because I can build useful automations quickly and only write a little code when I really need it. I like n8n because I can see the whole flow on one screen, test it node by node, and fix problems fast.

## Webhooks

A webhook is a way for one app to tell another app that something happened, right when it happens. Instead of my workflow asking again and again "is there anything new?", the other app sends a message to a special URL as soon as there is something. In n8n, a webhook node gives me that URL, and the workflow starts whenever data arrives. It works like a doorbell: the workflow waits quietly and wakes up when someone rings.

## REST APIs and HTTP requests

A REST API is a standard way for programs to talk to each other over the internet. I send an HTTP request to a URL, and the server sends a response back. The common request types are GET to read data, POST to send or create data, PUT to update and DELETE to remove. Requests often carry headers, such as an authorization key, and a body, and the answer usually comes back as JSON.

## JSON

JSON is the text format most APIs use for data. It stores information as key and value pairs inside curly braces, and lists inside square brackets. It is easy for people to read and easy for programs to parse, which is why almost every API and webhook uses it. A lot of my automation work is really about reading JSON from one step and reshaping it for the next step.

## Connecting everyday tools

I have worked with Google Sheets, Slack and Telegram in my workflows. Google Sheets can act as a simple database or a log of results. Slack and Telegram are useful for sending notifications or chatting with a bot. A small example of an idea: a workflow receives a message through a webhook, sends the text to an LLM through the OpenAI API, writes the result into a Google Sheet, and then notifies me on Telegram.

## AI inside automations

The OpenAI API lets a workflow use a language model as one of its steps. Prompt engineering is what makes that step reliable: I give the model a clear role, clear instructions and a clear output format. Adding RAG to such a workflow lets the model use my own documents, so the output is based on real information instead of guesses.

## Tips I follow

- Start with the simplest version of the workflow, then add steps.
- Test every node with real sample data.
- Keep API keys in credentials or secrets, never inside the workflow text or the code.
- Document what each workflow does, so it can be understood months later.
- Handle errors on purpose, because APIs sometimes fail.
