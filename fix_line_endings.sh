#!/bin/bash
# fix_line_endings.sh
# 用于将当前目录下所有 .sh 文件从 Windows 换行符转换为 Unix 格式

for file in *.sh; do
    if file "$file" | grep -q "CRLF"; then
        echo "正在处理 $file"
        sed -i 's/\r$//' "$file"
    fi
done

echo "换行符修复完成。"
