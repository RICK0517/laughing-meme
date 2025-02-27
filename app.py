def train_model(training_data, file_type):
    # 檢查數據格式
    if not training_data:
        st.error("訓練數據為空！")
        return

    # 解析數據
    texts = []
    if file_type == "json":
        try:
            data = json.loads(training_data)
            st.write(f"JSON 加載成功: {data}")  # Debug
            if not isinstance(data, list):
                st.error("JSON 格式錯誤，應該是列表")
                return
            texts = [item["text"] for item in data if "text" in item]
        except Exception as e:
            st.error(f"JSON 解析錯誤：{e}")
            return
    elif file_type == "txt":
        texts = [line.strip() for line in training_data.split("\n") if line.strip()]

    # Debug：確保 texts 正確
    st.write(f"處理後的文本數量：{len(texts)}")

    if not texts:
        st.error("沒有有效的文本數據！")
        return

    # 嘗試 Tokenization，增加錯誤處理
    try:
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256,  # 調整長度避免錯誤
            add_special_tokens=True
        )
    except Exception as e:
        st.error(f"Tokenization 失敗: {e}")
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
        st.write(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")

    # 保存模型
    model.save_pretrained(MODEL_SAVE_PATH)
    tokenizer.save_pretrained(MODEL_SAVE_PATH)
    st.success("模型訓練完成！")
