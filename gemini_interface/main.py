# import gradio as gr


# def greet(name):
#   return f"Hello , {name}!"

# demo = gr.Interface(greet, "text", "text")
# demo.launch(server_name="0.0.0.0")

import time
from typing import List, Tuple, Optional
import configparser
import os
import google.generativeai as genai
import gradio as gr
from PIL import Image

print("google-generativeai:", genai.__version__)

TITLE = """<h1 align="center">Gemini App</h1>"""
SUBTITLE = """<h2 align="center">Run with Gemini Pro Vision API</h2>"""

# 獲取當前腳本的目錄
script_dir = os.path.dirname(os.path.realpath(__file__))
# 使用相對路徑組合出配置文件的路徑
config_path = os.path.join(script_dir, "config.ini")

config = configparser.ConfigParser()
config.read(config_path)

# 從配置文件中讀取 Google API 金鑰
GOOGLE_API_KEY = config["DEFAULT"]["GOOGLE_API_KEY"]


AVATAR_IMAGES = (
    None,
    "image.jpg"
)


def preprocess_stop_sequences(stop_sequences: str) -> Optional[List[str]]:
    if not stop_sequences:
        return None
    return [sequence.strip() for sequence in stop_sequences.split(",")]


def user(text_prompt: str, chatbot: List[Tuple[str, str]]):
    return "", chatbot + [[text_prompt, None]]


def bot(
    #google_key: str,
    image_prompt: Optional[Image.Image],
    temperature: float,
    max_output_tokens: int,
    stop_sequences: str,
    top_k: int,
    top_p: float,
    chatbot: List[Tuple[str, str]]
):

    text_prompt = chatbot[-1][0]
    genai.configure(api_key=GOOGLE_API_KEY)
    generation_config = genai.types.GenerationConfig(
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        stop_sequences=preprocess_stop_sequences(stop_sequences=stop_sequences),
        top_k=top_k,
        top_p=top_p)

    if image_prompt is None:
        model = genai.GenerativeModel('gemini-1.5-flash-001')
        response = model.generate_content(
            text_prompt,
            stream=True,
            generation_config=generation_config)
        response.resolve()
    else:
        model = genai.GenerativeModel('gemini-1.5-flash-001')
        response = model.generate_content(
            [text_prompt, image_prompt],
            stream=True,
            generation_config=generation_config)
        response.resolve()

    # streaming effect
    chatbot[-1][1] = ""
    for chunk in response:
        for i in range(0, len(chunk.text), 10):
            section = chunk.text[i:i + 10]
            chatbot[-1][1] += section
            time.sleep(0.01)
            yield chatbot


image_prompt_component = gr.Image(type="pil", label="Image", scale=1)
chatbot_component = gr.Chatbot(
    label='Gemini',
    bubble_full_width=False,
    avatar_images=AVATAR_IMAGES,
    scale=2
)
text_prompt_component = gr.Textbox(
    placeholder="Hi there!",
    label="Ask me anything and press Enter"
)
run_button_component = gr.Button()
temperature_component = gr.Slider(
    minimum=0,
    maximum=1.0,
    value=0.4,
    step=0.05,
    label="Temperature",
    info=(
        "Temperature controls the degree of randomness in token selection. Lower "
        "temperatures are good for prompts that expect a true or correct response, "
        "while higher temperatures can lead to more diverse or unexpected results. "
    ))
        # "Temperature 控制令牌選擇的隨機程度 "
        # "較低的Temperature適用於期望獲得真實或正確回答的提示, "
        # "而較高的Temperature可以導致更多樣化或意外的結果 "
max_output_tokens_component = gr.Slider(
    minimum=1,
    maximum=2048,
    value=1024,
    step=1,
    label="Token limit",
    info=(
        "Token limit determines the maximum amount of text output from one prompt. A "
        "token is approximately four characters. The default value is 2048."
    ))
        # "Token 限制決定每個提示可以獲得的最大文字輸出量 "
        # "每個 Token 大約為四個字符，預設值為 2048 "
stop_sequences_component = gr.Textbox(
    label="Add stop sequence",
    value="",
    type="text",
    placeholder="STOP, END",
    info=(
        "A stop sequence is a series of characters (including spaces) that stops "
        "response generation if the model encounters it. The sequence is not included "
        "as part of the response. You can add up to five stop sequences."
    ))
        # "停止序列是一系列字元(包括空格),如果模型遇到它,會停止產生回應"
        # "此序列不作為反應的一部分，"
        # "可以增加多達5個停止序列"
top_k_component = gr.Slider(
    minimum=1,
    maximum=40,
    value=32,
    step=1,
    label="Top-K",
    info=(
       "Top-k changes how the model selects tokens for output. A top-k of 1 means the "
        "selected token is the most probable among all tokens in the model’s "
        "vocabulary (also called greedy decoding), while a top-k of 3 means that the "
        "next token is selected from among the 3 most probable tokens (using "
        "temperature)."
    ))
        # "Top-k 改變了模型為輸出選擇 token 的方式 "
        #  "Top-k 為 1 表示所選 token 在模型詞彙表中所有 token 中是最可能的(也稱為貪心解碼)"
        #  "而 top-k 為 3 意味著下一個 token 從最可能的 3 個 token 中選取(使用temperature)"
top_p_component = gr.Slider(
    minimum=0,
    maximum=1,
    value=1,
    step=0.01,
    label="Top-P",
    info=(
        "Top-k changes how the model selects tokens for output. A top-k of 1 means the "
        "selected token is the most probable among all tokens in the model’s "
        "vocabulary (also called greedy decoding), while a top-k of 3 means that the "
        "next token is selected from among the 3 most probable tokens (using "
        "temperature)."
    ))
         # "Top-p 改變了模型為輸出選擇 token 的方式 "
         # "token 從最可能到最不可能選擇,直到它們的機率總和等於 top-p 值 "
         # "如果 token A、B 和 C 的機率分別為 0.3、0.2 和 0.1,top-p 值為 0.5 "
         # "那麼模型會選擇 A 或 B 作為下一個 token(使用temperature) "
    

user_inputs = [
    text_prompt_component,
    chatbot_component
]

bot_inputs = [
    image_prompt_component,
    temperature_component,
    max_output_tokens_component,
    stop_sequences_component,
    top_k_component,
    top_p_component,
    chatbot_component
]

with gr.Blocks() as demo:
    gr.HTML(TITLE)
    gr.HTML(SUBTITLE)
    with gr.Column():
        with gr.Row():
            image_prompt_component.render()
            chatbot_component.render()
        text_prompt_component.render()
        run_button_component.render()
        with gr.Accordion("Parameters", open=False):
            temperature_component.render()
            max_output_tokens_component.render()
            stop_sequences_component.render()
            with gr.Accordion("Advanced", open=False):
                top_k_component.render()
                top_p_component.render()

    run_button_component.click(
        fn=user,
        inputs=user_inputs,
        outputs=[text_prompt_component, chatbot_component],
        queue=False
    ).then(
        fn=bot, inputs=bot_inputs, outputs=[chatbot_component],
    )

    text_prompt_component.submit(
        fn=user,
        inputs=user_inputs,
        outputs=[text_prompt_component, chatbot_component],
        queue=False
    ).then(
        fn=bot, inputs=bot_inputs, outputs=[chatbot_component],
    )

demo.queue(max_size=99).launch()
