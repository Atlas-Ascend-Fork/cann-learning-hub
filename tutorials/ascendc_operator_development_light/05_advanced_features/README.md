# 第5章 Ascend C算子进阶特性

## 章节概述
本章聚焦Ascend C算子的执行效率优化，帮助开发者理解Host侧重复下发开销的来源，掌握Aclgraph捕获、实例化与重放机制，并了解自定义算子通过Pybind11或Torch Library接入后被捕获和重放的过程。同时，面向Atlas A2/A3简要介绍多核并行、核间负载均衡、Double Buffer、UnitFlag、数据拌匀、UB Bank冲突、L2 Cache控制和多核同地址冲突，并提供对应的最佳实践样例链接。

## 在线体验
| Notebook | Link | 状态 |
|--|--|--|
| 5.1 章节介绍 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development_light&scanFilePath=tutorials/ascendc_operator_development_light/05_advanced_features/05.01_chapter_intro.ipynb) | ✅ 已发布 |
| 5.2 Aclgraph入图编译与运行 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development_light&scanFilePath=tutorials/ascendc_operator_development_light/05_advanced_features/05.02_aclgraph_compile_launch.ipynb) | ✅ 已发布 |
| 5.3 算子性能优化最佳实践 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development_light&scanFilePath=tutorials/ascendc_operator_development_light/05_advanced_features/05.03_best_practice.ipynb) | ✅ 已发布 |
