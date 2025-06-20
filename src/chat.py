from inference import ChatBot

model_path = "models/chatbot-epoch=09-val_loss=9.57.ckpt"
config_path = "src/config/params.yaml"

chatbot = ChatBot(model_path, config_path)

while True:
    question = input("You: ")
    if question.lower() in ("exit", "quit"):
        break
    response = chatbot.generate_response(question)
    print("Bot:", response)
