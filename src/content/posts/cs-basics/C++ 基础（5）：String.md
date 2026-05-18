---
title: C++ 基础（5）：String
published: 2026-05-03T09:13:37.731Z
description: ""
updated: ""
tags:
  - CPP
draft: false
pin: 0
toc: true
lang: ""
abbrlink: cpp-string
---
本文是对 [C++ 基础（3）：Streams](C++%20基础（3）：Streams.md) 的应用。
## 写：统一拼接并最后一次性拿到字符串

在工程里，如果需求只是**不断把不同类型的数据拼接起来，最后得到一个完整字符串**，那么最推荐使用 `std::ostringstream`。它的语义很明确：只负责写入，不负责解析。

```cpp
std::ostringstream os;

os << "this";
os << " ";
os << "is";

std::string data = os.str();
```

这里的 `os` 可以理解成一个“字符串构造器”。前面的 `<<` 都是在往内部缓冲区写内容，最后通过 `str()` 一次性得到完整的 `std::string`。

## 读：已经有字符串，现在要从里面解析内容

如果需求是**读取一个已有字符串中的内容**，例如按空格、逗号、换行解析，那么应该使用 `std::istringstream`

下面代码展示了根据空格拆分单词的例子：

```cpp
std::string data = "this is";

std::istringstream is(data);

std::string word;
while (is >> word) {
    std::cout << word << std::endl;
}
```

下面代码展示了把分隔符拼接的字符串还原成 token 的例子：

```cpp
#include <sstream>
#include <vector>
#include <string>

using namespace std;

// "1,2,#,4,#,#,3,#,#,"
// ->
// vector<string> nodes = {"1", "2", "#", "4", "#", "#", "3", "#", "#"};

vector<string> split(const string& data, char sep = ',') {
    vector<string> nodes;
    istringstream is(data);

    string token;
    while (getline(is, token, sep)) {
        nodes.push_back(token);
    }

    // 如果最后一个是空串，说明是 trailing delimiter
    if (!nodes.empty() && nodes.back().empty()) {
        nodes.pop_back();
    }

    return nodes;
}
```
