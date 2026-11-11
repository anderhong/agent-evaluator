import json
import time
import requests  # 🆕 需要 import requests

# ==========================================
# 1. 真實 Agent Call (透過 FastAPI)
# ==========================================
def call_real_agent(query):
    try:
        response = requests.post(
            "http://localhost:8000/invoke",
            json={"query": query},
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        return {
            "used_tool": data["used_tool"],
            "final_answer": data["final_answer"]
        }
    except Exception as e:
        return {
            "used_tool": "Error",
            "final_answer": f"API Error: {str(e)}"
        }

# ==========================================
# 2. Evaluation 主程式 (同之前一樣)
# ==========================================
def evaluate_agent(dataset_path):
    print(f"📥 Loading dataset from: {dataset_path}\n")
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    total_tests = len(dataset)
    correct_tool_count = 0
    keyword_match_count = 0
    total_latency = 0

    for test in dataset:
        query = test["query"]
        print(f"🧪 Testing: {query}")
        
        # 開始計時
        start_time = time.time()
        
        # Call 真實 Agent
        response = call_real_agent(query)
        
        # 計時結束
        latency = time.time() - start_time
        total_latency += latency
        
        # --- 評估 1：Tool Usage ---
        used_tool = response.get("used_tool", "None")
        expected_tool = test["expected_tool"]
        tool_correct = (used_tool == expected_tool)
        
        if tool_correct:
            correct_tool_count += 1
            tool_status = "✅"
        else:
            tool_status = f"❌ (Used: {used_tool}, Expected: {expected_tool})"
            
        # --- 評估 2：Answer Quality (Keywords) ---
        final_answer = response.get("final_answer", "").lower()
        expected_keywords = test["expected_keywords"]
        
        keywords_found = sum(1 for kw in expected_keywords if kw.lower() in final_answer)
        keyword_correct = (keywords_found == len(expected_keywords))
        
        if keyword_correct:
            keyword_match_count += 1
            answer_status = "✅"
        else:
            answer_status = f"❌ (Found {keywords_found}/{len(expected_keywords)} keywords)"
            
        # 印出結果
        print(f"   🔧 Tool: {tool_status}")
        print(f"   💬 Answer: {answer_status}")
        print(f"   ⏱️ Latency: {latency:.2f}s\n")

    # ==========================================
    # 3. 結算分數
    # ==========================================
    print("="*40)
    print("📊 Evaluation Summary")
    print("="*40)
    print(f"Total Tests: {total_tests}")
    print(f"🔧 Tool Accuracy:   {(correct_tool_count/total_tests)*100:5.1f}%  ({correct_tool_count}/{total_tests})")
    print(f"💬 Answer Accuracy: {(keyword_match_count/total_tests)*100:5.1f}%  ({keyword_match_count}/{total_tests})")
    print(f"⏱️ Avg Latency:     {total_latency/total_tests:5.2f}s")
    print("="*40)

if __name__ == "__main__":
    evaluate_agent("dataset.json")