# Changelog

所有对外可见的改动都记在这里。

**版本规则**：整个 workspace（`rui-api`、`rui-models`、`rui-ref`、`rui-expr`、`rui-nl`）共用一个版本号（根 `Cargo.toml` 的 `[workspace.package] version`，
代码中为 `rui_api::VERSION`）。0.0.x 阶段每一组合并进 `main` 的改动把补丁号加 1，合并后在 `main` 上打标签 `vX.Y.Z`。
`rui-api` 是求解器插件依赖的契约，它的不兼容改动在本文件里单独标出。

## [0.0.0] — 未发布

M0：契约、调度与参考求解器（设计见 `docs/design.md` §4）。

- **`rui-api`**（零依赖）：`Problem` + `SmoothOracle` / `BlackBoxOracle` 两种问题形态；`Signature` 与问题类别（LP … MINLP、BLACKBOX）；
  声明式的 `Capabilities` 与 `Verdict`（支持 / 只有点名时可用 / 不支持及原因）；`Options`（含文本解析）、`OptionSpec`、`ResolvedOptions`、
  通用选项；`Solution`（`Termination` 与 `PointKind` 两个维度，乘子约定 `σ∇f + Jᵀλ − z_L + z_U = 0`）；`Solver`、`AskTell`、`Context`、`Observer`；
  只用回调与原始边界的独立 KKT 检查 `kkt_check`。
- **`rui-models`**：`Registry`：校验、按 `PRIORITY` 自动调度或点名、桥接（最大化 → 最小化，点名黑箱求解器时光滑 → 黑箱）、选项解析
  （命名空间、别名、重复与未知选项报错并给出最接近的名字）、`explain`、ask/tell `Session`。
- **`rui-ref`**：参考求解器 `random`（随机搜索，混合变量、不等式约束、失败评估、ask/tell）与 `pgd`（投影梯度 + BB 步长 + Armijo 回溯，
  只处理界约束，收敛后用 `kkt_check` 核实乘子）；示例 `quickstart`。
- **`rui-expr`**：代数模型 `Model`（变量、优化方向、目标与约束 = 线性部分 + 表达式图上的节点；表达式图是操作数在前的 DAG，`Model::check` 校验）；
  按多项式次数给出函数类别（线性 / 二次 / 一般光滑，只往更一般的方向出错）；`ExprProblem`：精确的梯度、Jacobian、Lagrange Hessian，
  实现 `SmoothOracle`。求值器迁自私有 NLP 仓库的 `ampl-nlp-runner`（`eval.rs`），改动：目标按原样求值（最大化交给 `rui-models` 的桥接）、
  Jacobian 结构取线性变量与表达式变量之并、共享节点在一个元素内只编译一次。
- **`rui-nl`**：ASCII `.nl` 读取器（迁自同一处的 `nl.rs`，接受的子集不变），输出 `rui_expr::Model`；错误类型 `NlError`（读文件 / 语法）。
  在 Mittelmann 的 47 个 `.nl` 文件上，与原代码的结构和全部求值结果逐位一致。
- **CI**（GitHub Actions）：最新 stable 工具链；Ubuntu 上跑 `cargo fmt --check`、`clippy -D warnings`、`doc`（`RUSTDOCFLAGS=-D warnings`），
  Ubuntu / Windows / macOS 上跑 `cargo test --workspace --locked`，Ubuntu 上再跑 release 测试和 `quickstart` 示例。
- 测试 70 个：设计文档 §4.4 调度表的每一行（用与将来 nlp / bo 插件相同能力声明的 mock）、选项规则、桥接、校验、会话、中断与时间上限、端到端求解；
  模型校验、分类、共享节点、`.nl` 求值对照手算与有限差分、`.nl` 模型经 `Registry` 调度。
