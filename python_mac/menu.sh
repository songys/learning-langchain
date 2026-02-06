#!/usr/bin/env bash
set -euo pipefail

# ==================================================
#  LLM Pipeline Practice - One Click Runner
#  (macOS / Python 3.11)
# ==================================================

# Resolve repo root (this script is assumed at repo root)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

show_menu() {
    clear
    echo "=================================================="
    echo "  LLM Pipeline - One Click Runner"
    echo "  (macOS / Python 3.11)"
    echo "=================================================="
    echo ""
    echo "[ Environment Setup ]"
    echo "  1. Cloud API environment (GPT-5)"
    echo "  2. Local environment setup (DeepSeek-R1:14b)"
    echo ""
    echo "[ Run Experiments ]"
    echo "  3. GPT-5 No-RAG"
    echo "  4. GPT-5 Plain-RAG (Local Save)"
    echo "  5. GPT-5 Plain-RAG (LangSmith Tracking)"
    echo "  6. DeepSeek-R1 No-RAG"
    echo "  7. DeepSeek-R1 Plain-RAG"
    echo ""
    echo "  0. Exit"
    echo ""
}

while true; do
    show_menu
    read -rp "Select an option (0-7): " choice

    case "$choice" in
        0)
            echo "Bye!"
            exit 0
            ;;
        1)
            echo ""
            echo "[*] Setting up GPT-5 environment (py3.11)"
            bash env/mac/setup_py311.sh gpt5
            echo ""
            read -rp "Press Enter to continue..."
            ;;
        2)
            echo ""
            echo "[*] Setting up DeepSeek environment (py3.11)"
            bash env/mac/setup_py311.sh deepseek
            echo ""
            read -rp "Press Enter to continue..."
            ;;
        3)
            echo ""
            echo "[*] Running GPT-5 No-RAG experiment"
            bash env/mac/run_gpt5_no_rag.sh
            echo ""
            read -rp "Press Enter to continue..."
            ;;
        4)
            echo ""
            echo "[*] Running GPT-5 Plain-RAG experiment"
            bash env/mac/run_gpt5_plain_rag.sh
            echo ""
            read -rp "Press Enter to continue..."
            ;;
        5)
            echo ""
            echo "[*] Running GPT-5 Plain-RAG (LangSmith Tracking) experiment"
            bash env/mac/run_gpt5_plain_rag.sh --langsmith
            echo ""
            read -rp "Press Enter to continue..."
            ;;
        6)
            echo ""
            echo "[*] Running DeepSeek-R1 No-RAG experiment"
            bash env/mac/run_deepseekr1_no_rag.sh
            echo ""
            read -rp "Press Enter to continue..."
            ;;
        7)
            echo ""
            echo "[*] Running DeepSeek-R1 Plain-RAG experiment"
            bash env/mac/run_deepseekr1_plain_rag.sh
            echo ""
            read -rp "Press Enter to continue..."
            ;;
        *)
            echo ""
            echo "Invalid choice. Please try again."
            read -rp "Press Enter to continue..."
            ;;
    esac
done
