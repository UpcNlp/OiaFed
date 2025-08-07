#!/bin/bash
# FedCL 安装脚本
# 将 fedcl 命令安装到系统PATH中

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FEDCL_BIN="$SCRIPT_DIR/bin/fedcl"

echo "FedCL Installation Script"
echo "========================"

# 检查fedcl文件是否存在
if [ ! -f "$FEDCL_BIN" ]; then
    echo "❌ fedcl binary not found at: $FEDCL_BIN"
    exit 1
fi

# 确保fedcl有执行权限
chmod +x "$FEDCL_BIN"
echo "✅ Made fedcl executable"

# 提供安装选项
echo ""
echo "选择安装方式:"
echo "1. 创建符号链接到 /usr/local/bin (推荐)"
echo "2. 复制到 /usr/local/bin"
echo "3. 创建符号链接到 ~/.local/bin"
echo "4. 仅显示使用方法（不安装）"
echo "5. 退出"

read -p "请选择 (1-5): " choice

case $choice in
    1)
        TARGET="/usr/local/bin/fedcl"
        echo "🔗 创建符号链接到 $TARGET..."
        
        if [ -L "$TARGET" ] || [ -f "$TARGET" ]; then
            echo "⚠️  $TARGET 已存在"
            read -p "是否覆盖? (y/N): " overwrite
            if [ "$overwrite" != "y" ] && [ "$overwrite" != "Y" ]; then
                echo "❌ 安装取消"
                exit 1
            fi
            sudo rm -f "$TARGET"
        fi
        
        sudo ln -s "$FEDCL_BIN" "$TARGET"
        echo "✅ 符号链接创建成功!"
        ;;
        
    2)
        TARGET="/usr/local/bin/fedcl"
        echo "📄 复制到 $TARGET..."
        
        if [ -f "$TARGET" ]; then
            echo "⚠️  $TARGET 已存在"
            read -p "是否覆盖? (y/N): " overwrite
            if [ "$overwrite" != "y" ] && [ "$overwrite" != "Y" ]; then
                echo "❌ 安装取消"
                exit 1
            fi
        fi
        
        sudo cp "$FEDCL_BIN" "$TARGET"
        sudo chmod +x "$TARGET"
        echo "✅ 复制成功!"
        ;;
        
    3)
        LOCAL_BIN="$HOME/.local/bin"
        TARGET="$LOCAL_BIN/fedcl"
        
        # 创建目录（如果不存在）
        mkdir -p "$LOCAL_BIN"
        
        echo "🔗 创建符号链接到 $TARGET..."
        
        if [ -L "$TARGET" ] || [ -f "$TARGET" ]; then
            echo "⚠️  $TARGET 已存在"
            read -p "是否覆盖? (y/N): " overwrite
            if [ "$overwrite" != "y" ] && [ "$overwrite" != "Y" ]; then
                echo "❌ 安装取消"
                exit 1
            fi
            rm -f "$TARGET"
        fi
        
        ln -s "$FEDCL_BIN" "$TARGET"
        echo "✅ 符号链接创建成功!"
        
        # 检查PATH
        if [[ ":$PATH:" != *":$LOCAL_BIN:"* ]]; then
            echo ""
            echo "⚠️  $LOCAL_BIN 不在您的PATH中"
            echo "请将以下行添加到您的 ~/.bashrc 或 ~/.zshrc:"
            echo "export PATH=\"\$HOME/.local/bin:\$PATH\""
        fi
        ;;
        
    4)
        echo "📚 使用方法:"
        echo ""
        echo "直接使用完整路径:"
        echo "  $FEDCL_BIN run examples/config_templates/server_client_configs"
        echo ""
        echo "或者添加到PATH:"
        echo "  export PATH=\"$SCRIPT_DIR/bin:\$PATH\""
        echo "  fedcl run examples/config_templates/server_client_configs"
        echo ""
        echo "添加到shell配置文件（永久生效）:"
        echo "  echo 'export PATH=\"$SCRIPT_DIR/bin:\$PATH\"' >> ~/.bashrc"
        echo "  source ~/.bashrc"
        exit 0
        ;;
        
    5)
        echo "👋 退出"
        exit 0
        ;;
        
    *)
        echo "❌ 无效选择"
        exit 1
        ;;
esac

echo ""
echo "🎉 安装完成!"
echo ""
echo "验证安装:"
echo "  fedcl --version"
echo ""
echo "快速开始:"
echo "  fedcl run examples/config_templates/server_client_configs"
echo "  fedcl daemon examples/config_templates/server_client_configs"
echo "  fedcl status"
echo "  fedcl logs --follow"
echo "  fedcl stop"
echo ""
echo "更多帮助:"
echo "  fedcl --help"
