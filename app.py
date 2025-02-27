import streamlit as st
import torch
import json
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.utils.data import Dataset, DataLoader
import os

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
    def __init__(self, input_ids, attention_masks):
        self.input_ids = input_ids
        self.attention_masks = attention_masks

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_masks[idx]
        }

# 訓練模型（支持 JSON 或 TXT）
def train_model(training_data, file_type):
    if not training_data:
        st.error("訓練數據為空！")
        return

    texts = []
    if file_type == "json":
        try:
            data = json.loads(training_data)
            if isinstance(data, dict):
                data = data.get("texts", [])  # 支援 JSON 內含 "texts" 鍵
            texts = [item["text"] for item in data if "text" in item]
        except Exception as e:
            st.error(f"JSON 解析錯誤：{e}")
            return
    elif file_type == "txt":
        texts = [line.strip() for line in training_data.split("\n") if line.strip()]
    else:
        st.error("不支援的檔案類型！")
        return

    if not texts:
        st.error("解析後沒有可用的訓練數據！")
        return

    # 將文本轉換為 tokenized 格式
    encodings = tokenizer(
        texts,
        return_tensors="pt",
        max_length=512,
        truncation=True,
        padding="max_length",
        add_special_tokens=True
    )

    dataset = TextDataset(encodings["input_ids"], encodings["attention_mask"])
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    epochs = 3
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        st.write(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")

    model.save_pretrained(MODEL_SAVE_PATH)
    tokenizer.save_pretrained(MODEL_SAVE_PATH)
    st.success("模型訓練完成！")
    st.cache_resource.clear()

# 生成文本
def generate_text(prompt):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    output = model.generate(input_ids, max_length=100, num_return_sequences=1)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Streamlit UI
st.title("GPT-2 訓練與對話應用")
option = st.sidebar.selectbox("選擇功能", ["與 AI 對話", "上傳訓練檔案", "下載模型"])

if option == "與 AI 對話":
    st.header("與 AI 對話")
    user_input = st.text_input("輸入提示：")
    if st.button("生成"):
        if user_input:
            response = generate_text(user_input)
            st.write("AI 回應：")
            st.write(response)
        else:
            st.warning("請輸入內容！")

elif option == "上傳訓練檔案":
    st.header("上傳訓練檔案")
    uploaded_file = st.file_uploader("選擇檔案（JSON 或 TXT）", type=["json", "txt"])
    if uploaded_file:
        file_type = "json" if uploaded_file.type == "application/json" else "txt"
        content = uploaded_file.read().decode("utf-8")
        st.write("檔案解析成功！")
        if st.button("開始訓練"):
            train_model(content, file_type)

elif option == "下載模型":
    st.header("下載模型")
    if os.path.exists(MODEL_SAVE_PATH):
        st.download_button(
            "下載模型權重",
            data=open(os.path.join(MODEL_SAVE_PATH, "pytorch_model.bin"), "rb"),
            file_name="gpt2_finetuned.bin",
            mime="application/octet-stream"
        )
    else:
        st.warning("尚未訓練模型！")
