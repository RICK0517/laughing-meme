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
        
    # 設置填充符號（pad_token），如果沒有就使用 eos_token 或自定義填充符號
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # 使用 eos_token 來作為填充符號
    
    return tokenizer, model

tokenizer, model = load_model()

# 自定義 Dataset 類別
class TextDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}

# 訓練模型（支持 JSON 或 TXT）
def train_model(training_data, file_type):
    # 檢查數據格式
    if not training_data:
        st.error("❌ 訓練數據為空！")
        return

    # 解析數據
    texts = []
    if file_type == "json":
        try:
            data = json.loads(training_data)
            st.write(f"✅ JSON 加載成功: {data}")  # Debug
            if not isinstance(data, list):
                st.error("❌ JSON 格式錯誤，應該是列表")
                return
            texts = [item["text"] for item in data if "text" in item]
        except Exception as e:
            st.error(f"❌ JSON 解析錯誤：{e}")
            return
    elif file_type == "txt":
        texts = [line.strip() for line in training_data.split("\n") if line.strip()]

    # Debug：確保 texts 正確
    st.write(f"📊 處理後的文本數量：{len(texts)}")

    if not texts:
        st.error("❌ 沒有有效的文本數據！")
        return

    # 嘗試 Tokenization，增加錯誤處理
    try:
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,  # 確保填充
            max_length=256,  # 避免過長
            add_special_tokens=True
        )
    except Exception as e:
        st.error(f"❌ Tokenization 失敗: {e}")
        return

    # 創建 Dataset 和 DataLoader
    dataset = TextDataset(inputs)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # 訓練參數
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    epochs = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    # 訓練循環
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
        st.write(f"✅ Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")

    # 保存模型
    model.save_pretrained(MODEL_SAVE_PATH)
    tokenizer.save_pretrained(MODEL_SAVE_PATH)
    st.success("🎉 模型訓練完成！")

# 生成文本
def generate_text(prompt):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    output = model.generate(input_ids, max_length=100, num_return_sequences=1)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Streamlit UI
st.title("🤖 GPT-2 訓練應用")

# 側邊欄選單
option = st.sidebar.selectbox(
    "🔍 選擇功能",
    ["與 AI 對話", "上傳訓練檔案", "下載模型"]
)

if option == "與 AI 對話":
    st.header("🗣️ 與 AI 對話")
    user_input = st.text_input("🔹 輸入提示：")
    if st.button("🚀 生成"):
        if user_input:
            response = generate_text(user_input)
            st.write("🧠 AI 回應：")
            st.write(response)
        else:
            st.warning("⚠️ 請輸入內容！")

elif option == "上傳訓練檔案":
    st.header("📂 上傳訓練檔案")
    uploaded_file = st.file_uploader("📥 選擇檔案（JSON 或 TXT）", type=["json", "txt"])
    if uploaded_file:
        file_type = uploaded_file.type.split("/")[-1]
        content = uploaded_file.read().decode("utf-8")
        st.write("✅ 檔案解析成功！")
        if st.button("🚀 開始訓練"):
            train_model(content, file_type)

elif option == "下載模型":
    st.header("📥 下載模型")
    if os.path.exists(MODEL_SAVE_PATH):
        st.download_button(
            "⬇️ 下載模型權重",
            data=open(os.path.join(MODEL_SAVE_PATH, "pytorch_model.bin"), "rb"),
            file_name="gpt2_finetuned.bin",
            mime="application/octet-stream"
        )
    else:
        st.warning("⚠️ 尚未訓練模型！")
