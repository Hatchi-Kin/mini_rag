from fasthtml.common import *
import httpx
import asyncio

# Set up the app, including daisyui and tailwind for the chat component
tlink = Script(src="https://cdn.tailwindcss.com"),
dlink = Link(rel="stylesheet", href="https://cdn.jsdelivr.net/npm/daisyui@4.11.1/dist/full.min.css")
app = FastHTML(hdrs=(tlink, dlink, picolink), exts='ws')

# List of messages
messages = []

# Chat message component (renders a chat bubble)
def ChatMessage(msg):
    bubble_class = "chat-bubble-primary" if msg['role']=='user' else 'bg-indigo-700 text-white'
    chat_class = "chat-end" if msg['role']=='user' else 'chat-start'
    return Div(Div(msg['role'], cls="chat-header"),
               Div(msg['content'], cls=f"chat-bubble {bubble_class}"),
               cls=f"chat {chat_class}")

# The input field for the user message. Also used to clear the input field after sending a message via an OOB swap
def ChatInput():
    return Input(type="text", name='msg', id='msg-input',
                 placeholder="Type a message",
                 cls="input input-bordered w-full", hx_swap_oob='true')


# The main screen
@app.route("/")
def get():
    page = Body(H1('Chat with Megapi Documentation'),
                Div(*[ChatMessage(msg) for msg in messages],
                    id="chatlist", cls="chat-box h-[73vh] overflow-y-auto"),
                Form(Group(ChatInput(), Button("Send", cls="btn btn-primary")),
                    ws_send=True, hx_ext="ws", ws_connect="/wscon",
                    cls="flex space-x-2 mt-2",
                ),
                cls="p-4 max-w-lg mx-auto",
                )
    return Title('Megapi Doc'), page


@app.ws('/wscon')
async def ws(msg: str, send):
    # Send the user message to the user (updates the UI right away)
    messages.append({"role": "user", "content": msg.rstrip()})
    await send(Div(ChatMessage(messages[-1]), hx_swap_oob='beforeend', id="chatlist"))

    # Send the clear input field command to the user
    await send(ChatInput())
    # Simulate a delay
    await asyncio.sleep(.5)

    # Call the FastAPI endpoint to get the response
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post("http://localhost:7999/query", json={"query": msg.rstrip()})
            response.raise_for_status()
            response_data = response.json()
    except httpx.RequestError:
        response_data = {"result": "An error occurred while processing your request."}
    except httpx.HTTPStatusError:
        response_data = {"result": "An error occurred while processing your request."}

    # Get and send the model response
    if "result" in response_data and response_data["result"]:
        if isinstance(response_data["result"], dict):
            messages.append({"role": "assistant", "content": response_data["result"]["result"]})
        else:
            messages.append({"role": "assistant", "content": response_data["result"]})
    else:
        messages.append({"role": "assistant", "content": "No response received."})
    await send(Div(ChatMessage(messages[-1]), hx_swap_oob='beforeend', id="chatlist"))


if __name__ == '__main__':
    import uvicorn
    uvicorn.run("app:app", host='0.0.0.0', port=7998, reload=True)