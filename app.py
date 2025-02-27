import streamlit as st
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import os
from torch.utils.data import Dataset, DataLoader

# 設定模型保存路徑
MODEL_SAVE_PATH = "fine_tuned_gpt2"

# 載入或初始化模型和 tokenizer
@st.cache_resource
def load_model():
    if os.path.exists(MODEL_SAVE_PATH):
        tokenizer = GPT2Tokenizer.from_pretrained(MODEL_SAVE_PATH)
        model = GPT2LMHeadModel.from_pretrained(MODEL_SAVE_PATH)
    else:
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        model = GPT2LMHeadModel.from_pretrained("gpt2")
    return tokenizer, model

tokenizer, model = load_model()

# 自定義 Dataset 類別
class TextDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings.input_ids)

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}

# 訓練模型
def train_model(training_text):
    # 檢查 training_text 是否為空或格式不正確
    if not training_text or not isinstance(training_text, str):
        st.error("訓練數據為空或格式不正確！")
        return

    # 將文本轉換為 tokenized 格式
    inputs = tokenizer(training_text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    
    # 創建 Dataset 和 DataLoader
    dataset = TextDataset(inputs)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # 設定訓練參數
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    epochs = 3

    # 將模型移到 GPU（如果可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    # 訓練模型
    for epoch in range(epochs):
        for batch in dataloader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
        st.write(f"Epoch {epoch + 1} completed. Loss: {loss.item()}")

    # 保存模型
    model.save_pretrained(MODEL_SAVE_PATH)
    tokenizer.save_pretrained(MODEL_SAVE_PATH)
    st.success("模型訓練完成並已保存！")

# 生成文本
def generate_text(prompt):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    output = model.generate(input_ids, max_length=50, num_return_sequences=1)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

# Streamlit UI
st.title("GPT-2 訓練與對話應用程式")

# 側邊欄選單
option = st.sidebar.selectbox(
    "選擇功能",
    ["與 AI 對話", "上傳訓練用檔案", "下載訓練完畢的模型"]
)

if option == "與 AI 對話":
    st.header("與 AI 對話")
    user_input = st.text_input("輸入你的問題或提示：")
    if st.button("生成回應"):
        if user_input:
            response = generate_text(user_input)
            st.write("AI 的回應：")
            st.write(response)
        else:
            st.warning("請輸入一些文字！")

elif option == "上傳訓練用檔案":
    st.header("上傳訓練用檔案")
    uploaded_file = st.file_uploader("上傳訓練數據 (txt 檔案)", type="txt")
    if uploaded_file is not None:
        training_text = uploaded_file.read().decode("utf-8")
        st.write("檔案上傳成功！")
        if st.button("開始訓練"):
            train_model(training_text)

elif option == "下載訓練完畢的模型":
    st.header("下載訓練完畢的模型")
    if os.path.exists(MODEL_SAVE_PATH):
        st.write("訓練後的模型已準備好下載。")
        st.download_button(
            label="下載模型",
            data=open(os.path.join(MODEL_SAVE_PATH, "pytorch_model.bin"), "rb"),
            file_name="pytorch_model.bin",
            mime="application/octet-stream"
        )
    else:
        st.warning("尚未訓練模型，請先訓練模型！")
