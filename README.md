# Rui-Analytics-Models

Rui Analytics 的建模与调度层：问题描述一次，`solve` 按问题特征自动选一个求解器（也可以点名），选项、状态、结果的形式对所有求解器一致——
对应 SAS `PROC OPTMODEL` 里"一个模型 + `SOLVE [WITH 求解器] [/ 选项]`"的那部分能力；并通过开源工具已经在用的协议（AMPL `.nl` / `.sol`、
Python、Optuna、C API）让它们不必为 Rui 写专门的代码。设计见 [docs/design.md](docs/design.md)。

- 求解器（rui-nlp：内点法 NLP；rui-bo：贝叶斯优化）在各自的私有仓库里，通过很薄的插件 crate 实现本仓库公开的契约 `rui-api`；本仓库不依赖它们。
- 调度不夸大结果：KKT 点不叫全局最优，"局部不可行"不叫不可行性证明，黑箱优化的结果是"评估过的点里最好的"。
- 自动调度从不丢弃信息：有导数的问题只有在点名时才会交给黑箱求解器。

当前版本 **0.0.0**（[CHANGELOG.md](CHANGELOG.md)）。

## 状态

| 期 | 内容 | 状态 |
|---|---|---|
| M0 | 契约 `rui-api`、调度 `rui-models`、参考求解器 `rui-ref`、表达式层 `rui-expr` | 完成 |
| M1 | rui-nlp 插件、`.nl` / `.sol`、AMPL 驱动（Pyomo / JuMP / AMPL 接入） | `.nl` 读取器 `rui-nl` 已有（现有子集）；其余未开始 |
| M2 | Python 包 | 未开始 |
| M3 | rui-bo 插件、Optuna sampler | 未开始 |
| M4 | C API、Julia 原生封装 | 未开始 |

## 快速开始

开发与测试使用 Rust 1.100.0（`x86_64-pc-windows-gnu`）；没有外部依赖，可以离线构建。

```bash
cargo test --workspace
```

```bash
cargo run -p rui-ref --example quickstart
```

最小用法（Rust；求解器用的是公开仓库自带的参考求解器）：

```rust
use rui_api::{Options, Printer, ProblemRef};
use rui_models::Registry;

let registry = Registry::new().with(rui_ref::ProjectedGradient).with(rui_ref::RandomSearch);
// `problem` 实现 rui_api::SmoothOracle（有导数）或 rui_api::BlackBoxOracle（只能求值）
let r = registry.solve(ProblemRef::Smooth(&problem), &Options::parse("tol=1e-10")?, &mut Printer)?;
println!("{} | {}: x = {:?}, f = {:?}", r.termination, r.point, r.x, r.objective);
println!("{}", registry.explain(ProblemRef::Smooth(&problem), &Options::new()));   // 只看调度决定
```

`.nl` 文件（AMPL 格式）读成代数模型后，就是一个带精确一阶、二阶导数和函数类别的 `SmoothOracle`：

```rust
let model = rui_nl::read("problem.nl")?;                 // rui_expr::Model
let problem = rui_expr::ExprProblem::new(&model)?;       // 求值、求导、分类（LP / QP / QCQP / NLP）
let r = registry.solve(ProblemRef::Smooth(&problem), &Options::new(), &mut Printer)?;
```

读取器目前只接受一个子集（文本格式、无定义变量、无后缀、变量全连续），见 `crates/rui-nl/src/lib.rs` 开头；补全是 M1 的工作。

## 写一个求解器插件

实现 `rui_api::Solver`：`info()`（id、版本、`Capabilities`——能处理哪些变量、约束、函数类别，需要几阶导数）、`options()`（选项声明）、
`solve()`；黑箱求解器再实现 `session()` 返回 `AskTell`。Models 保证求解器拿到的是已校验的最小化问题、形态与能力声明相符、选项已校验；
求解器要在迭代之间检查 `Context::stop_reason()`、只经 `Context` 写日志。`crates/rui-ref/src/random.rs` 与 `pgd.rs` 是两个完整的样板。

## 目录结构

```
crates/rui-api/       契约：problem（变量、oracle）、signature（签名、能力、verdict）、options、solution、solver（Solver、AskTell、Context）、check（独立 KKT 检查）
crates/rui-models/    Registry：校验、调度（PRIORITY）、桥接（最大化、光滑→黑箱）、选项解析、explain、ask/tell Session
  tests/dispatch.rs   设计文档 §4.4 调度表的每一行，以及优先级、点名、桥接、校验
  tests/options.rs    选项解析规则与会话
crates/rui-ref/       参考求解器 random（随机搜索，带 ask/tell）与 pgd（投影梯度，只处理界约束）
  tests/end_to_end.rs 经 Registry 的端到端测试；examples/quickstart.rs
crates/rui-expr/      代数模型 Model（线性部分 + 表达式图）、按多项式次数分类、ExprProblem（精确导数，实现 SmoothOracle）
crates/rui-nl/        .nl 读取器（文本格式的一个子集）→ rui_expr::Model
  tests/eval.rs       求值与有限差分、元素共享、二次项、定义域错误
  tests/dispatch.rs   .nl 模型经 Registry 调度
docs/design.md        设计、分期与核实记录
```

## 许可证

尚未选择许可证（仓库所有者决定，见 [design.md §9](docs/design.md#9-需要你决定的事)）；在此之前各 crate 设为 `publish = false`。
