# Rui Analytics Models —— 设计（第 1 版）

状态：M0 已实现（契约、调度、参考求解器、表达式层 `rui-expr`）；`.nl` 读取器已迁入 `rui-nl`（现有子集），M1 其余部分待做 · 2026-11-19

Models 是 Rui Analytics 的建模与调度层：用户把问题描述一次，`solve` 按问题特征自动选一个 Rui 求解器（也可以点名），
选项、状态、结果的形式对所有求解器一致——对应 SAS `PROC OPTMODEL` 里"一个模型 + `SOLVE [WITH 求解器] [/ 选项]`"的那部分能力。
同时，Models 提供开源工具已经在用的几种接入协议，使 Pyomo、JuMP、AMPL、Optuna 等不需要为 Rui 写专门的代码就能调用到求解器。

## 0. 结论先行

1. **一个契约，两种问题形态。** 求解器只面对一个很小的公开契约（`rui-api`）：光滑问题走"二阶 oracle"（与 `rui_nlp::NlpProblem` 同形），
   黑箱问题走"零阶 oracle + ask/tell"（与 `rui_bo::Optimizer` 同形）。代数模型（表达式图）是这两者之上的一层，
   由 Models 负责分类、自动微分，再"降"成 oracle 交给求解器。
2. **调度是一张写死的表，不是启发式。** 先算出问题签名（变量类型、函数类别、可用的导数阶数、是否有界……），
   再按表选求解器；每次求解打印一行"问题类别 → 选了谁 → 为什么"。会丢信息的转换（例如把代数模型当黑箱）只在用户点名时才做。
3. **接入开源生态按投入产出排序**：① 符合 AMPL 约定的可执行文件（读 `.nl`、写 `.sol`）——Pyomo、JuMP、AMPL 一次打通；
   ② Python 包（建模 API + scipy / cyipopt 风格的函数接口 + ask/tell）；③ Optuna sampler（黑箱一侧的最大入口）；
   ④ C API（Julia 的 MathOptInterface 原生封装、C/C++、MATLAB 等的共同基础）。
4. **公开仓库不引用私有仓库。** 依赖方向倒过来：求解器仓库里各加一个很薄的插件 crate，实现公开的 `rui-api`；
   发布物（wheel、可执行文件、动态库）在一个私有的装配仓库里把"公开的 Models + 私有的插件"静态链接起来。
   这不是偏好，是实测结论：Cargo 即使在 feature 关闭、且提交了 `Cargo.lock` 的情况下，也会因为够不着的可选 git 依赖而构建失败（§10）。
5. **现成的资产直接复用。** 私有 NLP 仓库里 `ampl-nlp-runner` 的 `.nl` 读取器和自动微分求值器（约 2000 行）与求解算法无关，
   它的数据结构本身就是一个代数模型 IR。经你同意（§9 第 2 点），已迁到 Models：求值器在 `rui-expr`，读取器在 `rui-nl`；
   在 Mittelmann 的 47 个 `.nl` 文件上与原代码逐位一致（§10）。

## 1. 目标与非目标

**目标**

- G1 统一调度：同一个模型对象，`solve()` 自动选择，`solve("nlp")` / `solve("bo")` 点名；将来新增求解器（LP、QP、MILP……）时用户代码不变。
- G2 统一的选项、状态、结果，并且**不夸大**：沿用两个求解器现有的表述——KKT 点不叫全局最优，
  "局部不可行"不叫不可行性证明，贝叶斯优化的结果是"预算内观测到的最好点"。
- G3 开源程序低成本接入：优先实现它们已有的协议，而不是要求它们为 Rui 写插件。
- G4 公开仓库在没有任何私有代码的机器上可以独立构建、独立测试。

**非目标（第 1 版）**

- 不做独立的文本建模语言（OPTMODEL 那样的 `var / min / con` 语法）。建模 API 嵌在 Python 里；`.nl` 文件已经覆盖了"从别的建模语言来"的需求。
- 不调度第三方求解器。注册表的设计不排斥（一个可选的 Ipopt 参考插件对回归对比很有用），但不是目标。
- 不做模型层 presolve、多目标、问题修改后的热启动。
- 不提供与 `libipopt` 二进制兼容的"冒名"动态库（见 §5.5）。

## 2. 现状：设计必须遵守的事实

| | rui-nlp（0.1.3） | rui-bo（0.0.0） |
|---|---|---|
| 问题 | `min f(x)`，`g_L ≤ g(x) ≤ g_U`，`x_L ≤ x ≤ x_U`；光滑，可非凸；变量全连续 | `min f(x)`，可选 `c_k(x) ≤ 0`；变量为连续 / 整数 / 类别，可取对数尺度 |
| 需要的信息 | 函数值、梯度、Jacobian、Lagrange Hessian 的精确值（结构与数值分离） | 只要函数值；评估可以失败、可以带噪声 |
| 调用方式 | `Solver::new(opts).solve(&problem)`，求解器驱动回调 | `minimize(f, &space, &opts)`，或 `Optimizer` 的 `ask / ask_batch / tell / tell_failure` |
| 结果的含义 | 通过独立残差检查的一阶 KKT 点 | 预算内观测到的最好的可行点；带噪声时另有"推荐点" |
| 状态 | `SolveStatus`（15 种） | `Status`（4 种：预算用完、达到目标值、超时、有限空间耗尽） |
| 选项 | `SolverOptions` 结构体，字段名与 Ipopt 选项名一致（`tol`、`max_iter`、`mu_init`…） | `Options` 结构体 |
| 仓库 | 私有，Rust，无 `unsafe`，核心依赖只有 faer（可关） | 私有，Rust，无 `unsafe`，只依赖 `libm` |

由此得到的约束：

- 两个求解器的"问题"不是同一种东西：一个要导数、由求解器驱动；一个只要函数值、常常由外部驱动（并行试验、人工实验）。
  强行统一成一个 trait 会两头都别扭，所以契约里保留两种形态（§4.1）。
- 两者的"成功"也不是同一种东西：对 rui-bo，"预算用完"是正常结束。所以状态必须拆成"为什么停"和"手里的点是什么"两个维度（§4.6）。
- rui-nlp 目前必须有 Hessian（L-BFGS 未实现），没有热启动；只在迭代之间检查时间上限。只有一阶导数的问题目前接不进来。
- rui-bo 需要每个变量都有有限的界；状态不能序列化，但可以通过重新 `tell` 已有评估恢复。
- 两个求解器都纯 Rust、无本地依赖：Windows / Linux / macOS 的二进制分发没有额外障碍。
- `.nl` 读取器目前只接受自家翻译器写出的子集（文本格式、无定义变量、无后缀），而且没有 `.sol` 写出。

## 3. 总体结构

```
 谁在描述问题
 ┌────────────────┬─────────────────────┬──────────────────┬────────────────────────────┐
 │ rui.Model      │ 函数接口             │ 文件              │ 开源工具                    │
 │ （Python 建模） │ minimize / ask-tell │ .nl              │ Pyomo JuMP AMPL  │ Optuna  │
 └───────┬────────┴──────────┬──────────┴────────┬─────────┴────────┬─────────┴────┬────┘
         │ 表达式图           │ 回调               │ rui-nl           │ .nl / .sol    │ ask/tell
         ▼                   ▼                   ▼                  ▼ （rui 可执行文件）▼
 ┌──────────────────────────────────────────────────────────────────────────────────────┐
 │ rui-expr   代数模型 IR · 分类（线性 / 二次 / 一般非线性）· 自动微分求值                   │
 ├──────────────────────────────────────────────────────────────────────────────────────┤
 │ rui-models 问题签名 → 调度表 → 桥接 → 统一选项解析 → 结果回映；注册表；日志 / 中断 / 事件 │
 ├──────────────────────────────────────────────────────────────────────────────────────┤
 │ rui-api    契约：SmoothOracle · BlackBoxOracle · Solver · AskTell · Options · Solution  │   ← 以上全部公开
 └───────────────┬───────────────────────────────┬───────────────────────────────────────┘
 - - - - - - - - │ - - - - - - - - - - - - - - - │ - - - - - - - - - - - - - - - - - - - - -   私有
        rui-nlp-plugin → rui-nlp         rui-bo-plugin → rui-bo            （将来：lp / qp / milp …）
```

箭头只向下：求解器不知道 Python、`.nl`、Pyomo 的存在；入口不知道具体是哪个求解器。新增一个求解器 = 新增一个插件 crate + 调度优先级表里加一个 id。

## 4. 核心抽象（M0 已实现）

本节讲概念与理由；权威定义是代码：契约在 [`crates/rui-api`](../crates/rui-api/src/lib.rs)，调度在 [`crates/rui-models`](../crates/rui-models/src/lib.rs)。

### 4.1 问题的两种形态

**(a) 代数模型**（[`rui-expr`](../crates/rui-expr/src/lib.rs)，M0 已实现）：`Model` = 变量（`rui_api::Variable`：名字、类型、上下界、起点）、
优化方向、一个目标、若干约束 `g_L ≤ g(x) ≤ g_U`；每个函数 = 线性部分（按变量排序的稀疏系数）+ 可选的表达式（表达式图上的一个节点）。
这正是 `.nl` 文件的形状。约定与取舍：

- **表达式图是 DAG，操作数先于使用它的节点。** 节点可以被多个父节点、多个函数共享（将来 Python 建模 API 会大量共享）；
  `Model::check()` 校验这一点，于是分类和求值都可以按下标顺序一遍完成，不会遇到环。共享节点在一个元素内只编译一次。
- **分类按多项式次数，只往"更一般"的方向出错。** 每个节点算一个次数上界（常数 0、变量 1、乘积相加、常数非负整数次幂相乘，其余为"非多项式"），
  函数类别：≤ 1 线性，2 二次，其余一般光滑。`(x − x)·y` 会被当成二次，但一个非二次函数绝不会被标成二次——
  将来要求线性 / 二次的求解器不会拿到别的东西。
- **契约里没有"非光滑"类别。** 用到 `abs`、`min`、`max`、`floor`、`ceil`（参数非常数）的函数报成"一般光滑"：它们几乎处处可微，
  求值器给出当前分段的导数，与 AMPL 求解器库的做法一致。这是已知的不精确之处，要等契约增加类别时再改。
- **目标按原样求值，带着自己的优化方向。** 原代码在求值器里把最大化翻成最小化；迁入后改由 `rui-models` 的桥接统一处理（§4.4），
  求值器只报告 `Sense::Maximize`。
- **Jacobian 结构 = 线性部分的变量 ∪ 表达式里出现的变量。** `.nl` 的 `J` 段本来就列出全部变量，所以对 `.nl` 文件结构不变；
  别的前端不必为只在表达式里出现的变量填 0 系数。

`ExprProblem::new(&model)` 把模型降成 `SmoothOracle`：精确的一阶、二阶导数（`derivatives() = Hessians`），并通过 `function_classes()` 报告各函数的类别，
于是调度表能区分 LP / QP / QCQP / NLP。求值的做法（按元素拆分、相同元素去重、二次项闭式求导、元素内正向套反向求 Hessian）见
[`eval.rs`](../crates/rui-expr/src/eval.rs) 开头的说明。

**(b) oracle**（[`problem.rs`](../crates/rui-api/src/problem.rs)）：所有问题都实现 `Problem`（变量、约束上下界、优化方向），在此之上二选一：

| trait | 给出什么 | 同形于 |
|---|---|---|
| `SmoothOracle` | 函数值、梯度、Jacobian（结构与数值分离），可选 Lagrange Hessian；可选声明各函数的类别（线性 / 二次 / 一般光滑） | `rui_nlp::NlpProblem`、Ipopt 的 TNLP |
| `BlackBoxOracle` | 一次评估给出目标和全部约束值，可以失败（`Outcome::Failed`），可以声明带噪声 | `rui_bo::minimize_outcome` 的回调 |

约定：0 起索引；无界用 ±∞；点是 `Vec<f64>`（整数变量取整数值，类别变量取水平的下标）。
`SmoothOracle::derivatives()` 默认只有一阶导数，提供 Hessian 的问题必须明确声明——漏了声明，后果只是需要 Hessian 的求解器拒绝它，
而不是求解途中才出错。用户直接写回调（Python 函数、Rust 闭包）时，跳过代数层，直接实现 oracle。

### 4.2 问题签名

`Signature`（[`signature.rs`](../crates/rui-api/src/signature.rs)）是纯数据：各类变量个数、无界变量个数、约束个数与等式个数、
目标与各约束的类别、可用的导数（`Values < Gradients < Hessians`）、优化方向、是否带噪声。
由它得到问题类别 `LP / QP / QCQP / NLP / MILP / MIQP / MIQCQP / MINLP / BLACKBOX`，用于日志和报错。

与草案的差别：去掉了 `Access::Algebraic`。表达式带来的信息在调度上只体现为"函数类别已知"，所以改由 `SmoothOracle::function_classes()` 表达；
不声明类别的光滑问题一律按 NLP 处理。

### 4.3 求解器契约

```rust
pub trait Solver: Send + Sync {
    fn info(&self) -> SolverInfo;                         // id、名称、版本、说明、Capabilities
    fn options(&self) -> Vec<OptionSpec>;                 // 选项声明
    fn check(&self, sig: &Signature) -> Vec<String>;      // Capabilities 表达不了的限制；默认没有
    fn solve(&self, problem: ProblemRef<'_>, options: &ResolvedOptions, ctx: &mut Context<'_>) -> Solution;
    fn session(&self, space: &dyn Problem, options: &ResolvedOptions) -> Result<Box<dyn AskTell>, Error>;  // 默认不支持
}

pub trait AskTell: Send {   // ask(q) · tell(x, outcome) · mark_pending(x) · solution() · n_evaluations()
}
```

- **能力是数据，不是代码。** 求解器在 `Capabilities` 里声明：使用哪一级信息（`uses`）、目标与约束最多到哪一类函数、是否接受整数 / 类别变量、
  是否要求所有变量有有限的界、是否接受不等式 / 等式约束、是否有 ask/tell。`Capabilities::verdict(&Signature)` 对所有求解器用同一套规则给出
  `Supported`、`Lossy`（只有丢掉导数、把问题当黑箱才能解）或 `Unsupported(原因列表)`。这样原因的措辞一致，`explain` 能打印，
  插件作者不经过 Models 就能单测自己的能力声明。
- **Models 向求解器保证**：问题已校验（上下界一致、起点在界内、整数变量的界是整数）；一律是**最小化**问题；`uses == Values` 的求解器拿到
  `BlackBoxOracle`，否则拿到 `SmoothOracle`；选项已按声明校验并填好默认值。
- **求解器向 Models 保证**：至少在迭代 / 评估之间检查 `Context::stop_reason()`（中断与时间上限）；只经过 `Context` 写日志；不夸大结果（§4.6）。
- `Context` 持有一个 `Observer`，负责三件事：日志行；进度事件（`Iteration` / `Evaluation`，返回 `Flow::Stop` 可以停止求解）；
  `interrupted()`（Ctrl-C 之类）。中断并进 `Observer`，所以 `solve` 只需要一个 `observer` 参数。

### 4.4 调度

（[`lib.rs`](../crates/rui-models/src/lib.rs)、[`dispatch.rs`](../crates/rui-models/src/dispatch.rs)）

```
solve(problem, options, observer):
    校验 → 签名 → 每个求解器的 verdict → 选择 → 解析选项 → 桥接 → 求解 → 结果回映
选择：点名（method=id）：该求解器 Supported 或 Lossy 即可；Unsupported 报错并给出原因
      自动：按 PRIORITY 顺序取第一个 Supported 的（Lossy 不算）；没有就报错，逐个列出原因，并指出哪些求解器"点名即可用"
```

`PRIORITY = ["nlp", "bo", "pgd", "random"]`；不在表里的 id 排在后面，按注册顺序。注册顺序不影响选择（有测试）。
每次求解在 `Solution::notes[0]` 记一行，并按 `verbosity` 输出，例如
`rui: NLP (n=2, m=0) -> pgd 0.0.0 (automatic)`、`rui: NLP (n=2, m=1) -> random 0.0.0 (named; derivatives ignored: solved as a black box)`。

生产组合（nlp + bo）下的调度表。每一行在 [`tests/dispatch.rs`](../crates/rui-models/tests/dispatch.rs) 里有对应的测试，
测试用的 mock 求解器声明的能力与将来的 nlp / bo 插件相同：

| 签名 | 自动选择 | 测试 |
|---|---|---|
| 光滑、变量全连续、有二阶导数（含 LP / QP，直到有专门的求解器） | `nlp` | `row1_*` |
| 黑箱、变量全部有界（可含整数 / 类别） | `bo` | `row2_*` |
| 黑箱、有变量无界 | 报错：bo 需要有限的界 | `row3_*` |
| 光滑但只有一阶导数 | 报错；可以点名 `bo` | `row4_*` |
| 含整数或类别变量的代数模型 | 报错；可以点名 `bo` | `row5_*` |
| 黑箱含等式约束 | 报错：bo 不处理等式 | `black_box_with_an_equality_constraint_has_no_solver` |

公开仓库只有参考求解器时：只有界约束、有梯度的光滑问题 → `pgd`；有界黑箱 → `random`；带一般约束的光滑问题报错，可以点名 `random`。
`.nl` 模型走的是同一条路径（[`rui-nl/tests/dispatch.rs`](../crates/rui-nl/tests/dispatch.rs)）：只有界约束的二次模型被识别为 `QP (n=2, m=0)` 并交给 `pgd`；
最大化模型经桥接求解、结果按原方向报告；带线性约束的模型自动调度报错，点名 `random` 后可解。
发布版注册哪些求解器由装配仓库决定；建议注册 `nlp`、`bo` 和 `random`（作为基线），不注册 `pgd`。

已实现的桥接（[`bridge.rs`](../crates/rui-models/src/bridge.rs)）：
- 最大化 → 最小化：目标与梯度取负，Hessian 的 `obj_factor` 取负；进度事件里的目标值再取负还原，返回结果的目标值还原，乘子不变（§4.6）。
  求解器自己写的日志行显示的是被最小化的目标值。
- 点名黑箱求解器时，光滑 → 黑箱：求值出错算一次失败的评估。

约束 `g_L ≤ g ≤ g_U` 不做转换，黑箱求解器自己按区间判断可行性（`rui_api::constraint_violation`）。

`Registry::explain` 只做校验、分类和调度，不求解。参考求解器上的输出（来自 `cargo run -p rui-ref --example quickstart`）：

```
problem  NLP: 2 variables (2 continuous), 0 constraints; objective smooth nonlinear; first derivatives; minimize
solvers  in priority order
  pgd     0.0.0     supported
  random  0.0.0     only when named: ignores the derivatives and treats the problem as a black box
choice   rui: NLP (n=2, m=0) -> pgd 0.0.0 (automatic)
```

### 4.5 选项

（[`options.rs`](../crates/rui-api/src/options.rs)、[`resolve.rs`](../crates/rui-models/src/resolve.rs)）

- **全局选项**：`method`（由调度使用；缺省或 `auto` 即自动选择）、`time_limit`、`verbosity`。每次求解都接受，由 Models 通过 `Context` 施加，每个求解器都必须遵守。
- **可选的通用选项**：`max_iterations`、`max_evaluations`、`tol`、`seed`。求解器用 `rui_api::common` 里的构造函数声明自己支持的那几个，
  名字、类型、说明因此一致；被选中的求解器不支持时，忽略并提示一次。
- **命名空间**：`nlp.mu_init` 只在 `nlp` 被选中时生效，否则忽略并提示；命名空间不是已注册的求解器就报错。
  OPTMODEL 规定只有写了 `WITH` 才能给求解器选项，命名空间去掉了这个限制。
- **别名**：声明可以带别名，例如 `max_iterations` 的别名 `max_iter`，让 Pyomo / JuMP 用户写惯的 Ipopt 选项名原样可用。
  同一个选项给了两次（两个别名，或一次带命名空间一次不带）报错。
- **未知的键**：不带命名空间、被选中的求解器不认识的键，如果属于另一个已注册的求解器，报错并建议写成 `other.key`；
  否则报"未知选项"，并给出编辑距离最近的名字（相邻字符对调算一次编辑，所以 `tlo` 会提示 `tol`）。从不静默忽略。
- **文本形式**：`Options::parse("tol=1e-8 method=nlp name=\"a b\" flag")`。值先存为文本，解析时按声明的类型转换并检查范围
  （`1e3` 可以作整数，`inf` 可以作数值，`yes/no/on/off/1/0` 可以作布尔，选择型选项不区分大小写）。
- 点名求解器的键叫 `method`，不叫 `solver`：Pyomo 的通用 ASL 插件把 `options['solver']` 保留为可执行文件名，不会传给求解器（§10）。
- 求解器拿到的 `ResolvedOptions` 里每个值都符合声明；`f64() / i64() / usize() / bool() / text()` 读取声明过的选项，读未声明的选项会 panic——
  那是插件的 bug，测试即可发现。注册时也校验选项声明（名字合法、互不重复、不与全局选项冲突、默认值符合类型），出错直接 panic。

### 4.6 结果与状态

（[`solution.rs`](../crates/rui-api/src/solution.rs)、[`check.rs`](../crates/rui-api/src/check.rs)）

`Solution`：`termination`（为什么停）、`point`（返回的点是什么）、`message`、`x`、`objective`（按原问题的方向）、`constraints`、`duals`、
`stats`、`notes`（调度行与选项提示）、`native`（求解器自己的结果：迭代历史、全部评估记录等）。

| `PointKind` | 含义 | 来自 |
|---|---|---|
| `KktPoint { verified }` | 满足容差的一阶 KKT 点；非凸时只是局部驻点。`verified` = 通过只用回调与原始边界的独立残差检查 | nlp：`Solved`、`SolvedToAcceptableLevel`；pgd：收敛 |
| `BestObserved { feasible }` | 评估过的点里最好的；没有最优性证书 | bo、random |
| `InfeasibleStationary` | 不可行度的驻点；不是不可行性证明 | nlp：`LocallyInfeasible` |
| `LastIterate` | 停下时的迭代点，没有任何保证 | 各种上限与失败 |
| `NoPoint` | 没有可用的点 | 起点评估失败、时间上限为 0 等 |

与草案的差别：`LocalOptimum` 改名为 `KktPoint`，因为 KKT 点可能是鞍点，"局部最优"会夸大；`Unverified` 改名为更直白的 `LastIterate`。

`Termination`：`Converged`、`ConvergedAcceptable`、`LocallyInfeasible`、`Diverging`、`IterationLimit`、`EvaluationLimit`、`TimeLimit`、
`TargetReached`、`SpaceExhausted`、`AskTell`、`SlowProgress`、`SolverFailure`、`NumericalFailure`、`EvaluationError`、`CheckFailed`、
`Interrupted`、`InvalidProblem`。插件用穷尽的 `match`（不写 `_ =>`）把原生状态映射过来，漏掉的状态在编译时就会暴露。
`Solution::is_success()` 的定义：`KktPoint`，或 `BestObserved { feasible: true }`。

**错误与结果分开。** 没有求解（问题或选项无效、没有合适的求解器、点名的求解器不支持）是 `Err(rui_api::Error)`；
求解器自己报告的一切，包括失败，都是 `Ok(Solution)`。AMPL 驱动会把 `Err` 写成 500 段的 `solve_result_num`。

**乘子约定。** 令 σ = +1（最小化）或 −1（最大化），则 `σ∇f + Jᵀλ − z_L + z_U = 0`，`z_L, z_U ≥ 0`；
只有下侧可能起作用的约束 `λ ≤ 0`，只有上侧的 `λ ≥ 0`，等式不限符号。这是被最小化的函数 σf 的 Ipopt 约定，
好处是最大化问题经过桥接后乘子不必变号，符号条件也与优化方向无关。
`rui_api::kkt_check(problem, x, duals)` 只用回调和原始边界计算原始不可行度、对偶不可行度、互补性和符号违反，全部是不缩放的无穷范数。
互补性取 `min(|松弛量|, |乘子|)`：两者之一为 0 时才为 0，对大乘子不像乘积那样敏感。NaN 一律算作无穷大。

**向外部生态映射（M1）时取对方词汇里最弱的真话。** `.sol` 的 `solve_result_num` 按区间被读取（MathOptInterface 的读取器已核对，§10）：

| Termination | `solve_result_num` | JuMP 经 AmplNLWriter 看到的 |
|---|---|---|
| `Converged` | 0–99 | `LOCALLY_SOLVED` |
| `ConvergedAcceptable` | 100–199 | `LOCALLY_SOLVED` |
| `LocallyInfeasible` | 200–299 | `LOCALLY_INFEASIBLE` |
| 各种上限 | 400–499 | `OTHER_LIMIT` |
| 各种失败、没有合适的求解器 | 500–599 | `OTHER_ERROR` |

对偶变量的符号约定在每个适配器里各不相同（AMPL、Pyomo、JuMP 对最大化问题和对 `≤ / ≥` 约束的规定不一样），是这类接口最常见的错误来源。
规则：`Solution` 内部只用上面那一种约定；每个适配器写明自己的换算，并用"同一个模型、同一条路径、与 Ipopt 返回的对偶值逐项比较"来验收。

## 5. 入口：开源程序怎样用到 Rui

### 5.1 符合 AMPL 约定的可执行文件 `rui`（第一优先）

一个可执行文件，三家建模工具零代码接入：

```python
# Pyomo —— 未注册的求解器名会落到通用 ASL 插件，把名字当作可执行文件
pyo.SolverFactory("rui").solve(model, options={"tol": 1e-8})
```

```julia
# JuMP
model = Model(() -> AmplNLWriter.Optimizer("rui"))
```

```ampl
option solver rui;  option rui_options 'tol=1e-8 method=nlp';  solve;
```

需要实现的约定（括号里是依据）：

- 调用形式 `rui <stub 或 stub.nl> -AMPL [key=value …]`，结果写到同名的 `.sol`（Pyomo：`[可执行文件, 问题文件, '-AMPL']` 加选项；AmplNLWriter：`solver model.nl -AMPL 选项…`，读回 `model.sol`）。
- 选项同时从环境变量 `rui_options` 和命令行 `key=value` 读取，命令行优先（Pyomo 两处都给；AmplNLWriter 只给命令行；AMPL 只给环境变量）。
- `rui -v` 必须在几秒内打印可解析的版本号：Pyomo 以此判断求解器是否可用（`[solver_exec, "-v"]`，超时 5 秒）。
- `.sol` 里写原始解、约束对偶、`solve_result_num` 和消息；界乘子写成后缀 `lower_bound_duals` / `upper_bound_duals`
  （MathOptInterface 默认认这两个名字，也认 `ipopt_zL_out` / `ipopt_zU_out`；加一个选项可以同时写 Ipopt 的名字，让现有 Pyomo 脚本不用改）。
- `.nl` 读取器要补的部分，按 Pyomo 的 NL writer 实际写出的内容确定：`V` 段（定义变量——Pyomo 默认把具名 `Expression` 写成 `V` 段，
  现有读取器会拒绝）、`S` 段（后缀）、`x` / `d` 段（原始 / 对偶起点；`d` 在有热启动之前读入后忽略）、多个目标（取第一个或按 `objno`）、
  `r` 段类型 5（互补）与 `F` 段（外部函数）明确报错。Pyomo 只写文本格式（`g3`）；二进制格式留到正式支持 AMPL 时再做。
- **界外起点（M1 要先决定）。** AMPL 允许 `x` 段的起点落在变量的界外，求解器自己把它移进来；`rui-nl` 按原样读入。
  但 `rui-models` 的校验目前拒绝界外起点，于是 Mittelmann 47 题里有 4 题经 `Registry` 会被拒：WM_CFy（756 个变量）、arki0003（31）、
  arki0009（9）、camshape_6400（2）。建议：连续变量的界外起点由 Models 投影到界内，并在 `Solution::notes` 里记一条说明（不悄悄改）；
  采用前要核对 rui-nlp 对"投影后的起点"与"原起点"算出的初始点相同，否则 runner 与驱动的结果会不一致。离散变量的界外起点仍报错。
- `.nl` 是代数模型，所以走完整的调度：全连续 → `nlp`；头部声明了整数变量 → 报"没有对应求解器"并写出 500 段的 `solve_result_num`。
  将来有了 LP / MILP 求解器，同一个可执行文件、同样的用户脚本直接受益。

同一个可执行文件也是命令行工具：`rui solve file.nl`、`rui explain file.nl`、`rui solvers`、`rui options nlp`。

### 5.2 Python 包

```python
import rui

m = rui.Model("hs071")
x = m.var(4, lb=1, ub=5, start=[1, 5, 5, 1])
m.minimize(x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2])
m.subject_to(x[0] * x[1] * x[2] * x[3] >= 25, name="prod")
m.subject_to(rui.sum(x**2) == 40)

r = m.solve()                              # 全连续、有表达式 → 自动选 nlp
r = m.solve("nlp", tol=1e-10, options={"nlp.mu_init": 1e-2})
print(r.termination, r.point, r.objective, x.value, m.dual("prod"))
```

```python
m = rui.Model()
lr     = m.var(lb=1e-4, ub=1e-1, log=True)
layers = m.var(lb=1, ub=8, integer=True)
act    = m.var(choices=["relu", "tanh", "gelu"])
m.minimize(train_and_validate, args=[lr, layers, act])   # 目标是 Python 函数 → 零阶 oracle → 自动选 bo
r = m.solve(max_evaluations=60)
```

同一个 `Model`，目标是表达式还是不透明的函数，决定了走哪个求解器——这是 OPTMODEL 式调度的核心。在它旁边提供三类不需要建模的接口：

- **函数接口**：`rui.minimize(fun, x0, jac=, hess=, bounds=, constraints=)`，参数形式跟 `scipy.optimize.minimize` 一致；
  `rui.minimize(fun, space=…)` 用于黑箱。
- **ask/tell**：`rui.Optimizer(space, **options)`，方法与 rui-bo 的 `Optimizer` 一一对应。
- **兼容层**（各自独立的子模块，不引入硬依赖）：`rui.compat.cyipopt` 提供与 `cyipopt.Problem`（`objective / gradient / constraints / jacobian / hessian`，
  可选 `jacobianstructure / hessianstructure`）同形的类，让按 cyipopt 写的代码改一行 import 就能换求解器。

实现：PyO3 + maturin，abi3 wheel（每个平台一个 wheel 覆盖多个 Python 版本）；§5.1 的可执行文件一起装进 wheel，
于是 `pip install` 之后 Pyomo 的 `SolverFactory("rui")` 立即可用。代数模型在 Rust 里求值和求导，求解过程中不回调 Python；
只有用户给的是 Python 函数时才在每次评估时回到解释器。

### 5.3 Optuna sampler（黑箱一侧的第一优先）

`rui.integrations.optuna.RuiSampler(BaseSampler)`：

- `infer_relative_search_space`：用 Optuna 推荐的交集搜索空间；`Float / Int / Categorical` 分布（含 `log`）对应 rui-bo 的五种变量。
- `sample_relative`：按 study 里的试验重建会话——`COMPLETE` → `tell`，`FAIL` → `tell(…, Failed)`，`RUNNING` → `mark_pending`——然后 `ask`。
  Optuna 的 sampler 在多进程之间不共享内存状态，rui-bo 又恰好支持"重新 `tell` 恢复"，两者正好对上。
- `sample_independent`：相对搜索空间之外的参数（主要是第一个试验）退回随机采样。

其他试验调度框架（Ray Tune 等）有需求时照这个模式加，都落在 `AskTell` 上。

### 5.4 C API（基础设施，放在后面）

`rui.h` + 动态库，由 `rui-capi` 从同一套 Rust 接口导出：`rui_problem_*`（回调函数指针 + 用户数据，形状参照大家熟悉的 Ipopt C 接口，
便于移植现有的封装）、`rui_options_set(name, value)`、`rui_solve`、`rui_session_ask / tell`。它是以下各项的共同前提：
Julia 包 `Rui.jl`（MathOptInterface 原生封装：不经文件、可设回调、对偶值直接返回）、C / C++ 用户、MATLAB、R。
放在后面是因为 §5.1 已经让 JuMP 可用；原生封装改善的是体验，不是"能不能用"。

### 5.5 不做的事

- **与 `libipopt` 二进制兼容的替身库**：能让所有基于 Ipopt 的程序不改代码就换求解器，但对方期待的是 Ipopt 的全部选项和行为，
  任何差异都会变成难以定位的问题，而且会让用户分不清自己在用什么。用显式的适配器（§5.2 的 `compat.cyipopt`、§5.4 的 C API）代替。
- **先为 Pyomo / JuMP 写原生插件**：`.nl` 路线已经覆盖；原生插件等 C API 稳定后再做。

## 6. 仓库布局与发布

```
Rui-Analytics-Models（公开）
  crates/
    rui-api/       契约：trait 与纯数据类型，零依赖。求解器仓库唯一需要依赖的 crate          【M0 已有】
    rui-models/    注册表、调度、桥接、选项解析、结果回映、explain、ask/tell 会话             【M0 已有】
    rui-ref/       参考求解器：random（随机搜索，带 ask/tell）、pgd（投影梯度，只处理界约束）  【M0 已有】
    rui-expr/      代数模型 IR、分类、自动微分求值（迁自 ampl-nlp-runner 的 eval.rs）         【M0 已有】
    rui-nl/        .nl 读取（迁自 nl.rs，现有子集）、.sol 写出                              【读取器已有；补段与 .sol 在 M1】
    rui-driver/    命令行与 AMPL 驱动（库）：run(registry, argv)                            【M1】
    rui-python/    PyO3 绑定（库）：register(module, registry)                             【M2】
    rui-capi/      C API                                                                  【M4】
  python/rui/      纯 Python 部分：建模语法、compat、integrations                          【M2】
  docs/  tests/

Rui-Analytics-NLP-Solver（私有）      + crates/rui-nlp-plugin   实现 rui_api::Solver
Rui-Analytics-Bayesian-Solver（私有） + crates/rui-bo-plugin    实现 rui_api::Solver 与 AskTell
Rui-Analytics-Dist（私有，新建）        三个各约 20 行的入口，把公开库和私有插件链接成发布物
```

```rust
// Rui-Analytics-Dist 里的可执行文件入口，全部内容：
fn main() {
    let registry = rui_models::Registry::new()
        .with(rui_nlp_plugin::solver())
        .with(rui_bo_plugin::solver())
        .with(rui_ref::RandomSearch);
    std::process::exit(rui_driver::run(registry, std::env::args()));
}
```

- **依赖方向**：`rui-nlp`、`rui-bo` 本身不变、不依赖 Models；插件 crate 依赖 `rui-api`；Dist 依赖所有人。Models 的任何 crate 都不出现私有仓库的地址。
- **公开仓库自带可运行的系统**：用 `rui-ref` 的参考求解器，调度、桥接、选项、会话在公开测试里端到端可测（M0 已做到）；
  M1 起 `rui-driver` 和 Python 包也照此测试。参考求解器同时是"怎样写一个插件"的样板。
- **版本**：`rui-api` 独立做语义化版本；因为是静态链接，契约不兼容在编译时暴露，而不是在用户机器上。一次发布 = (Models, nlp, bo) 三个版本的组合，
  `rui -v` 全部打印，Dist 的 `Cargo.lock` 记录确切的提交。
- **本机开发**：Dist 里用 `[patch]` 指向同级目录的三个检出，改任何一边都立即生效。
- **适配代码跟着求解器走**：选项表、状态映射在插件 crate 里，与求解器在同一个仓库、同一次提交里改，不会两边脱节。

## 7. 需要求解器仓库配合的改动

都很小，且不触及算法：

| 仓库 | 改动 | 原因 |
|---|---|---|
| 两者 | 日志经一个可替换的输出口，而不是直接 `println!`（rui-nlp 源码里 11 处，rui-bo 7 处） | Jupyter、日志文件、AMPL 驱动各有去处 |
| 两者 | 新增插件 crate：`Capabilities`、选项声明（名字 → 字段）、状态映射 | §4.3 |
| rui-nlp | 一次因子分解期间也能响应中断 / 时间上限 | 现在只在迭代之间检查 |
| rui-bo | `Optimizer` 增加"登记一个别处正在评估的点"的方法 | 现在挂起点只能由 `ask` 产生；`AskTell::mark_pending` 与 Optuna 多 worker 需要 |
| NLP 仓库 | `ampl-nlp-runner` 的 `nl.rs`、`eval.rs` 迁到 Models；runner 改为依赖 Models 的 crate | §0 第 5 点。Models 一侧已完成，与原代码逐位一致（§10）。runner 一侧待做：删掉两份副本，依赖 `rui-nl` + `rui-expr`，在它的 `NlpProblem` 适配器里自己处理最大化（目标、梯度乘 −1，Hessian 的 `obj_factor` 乘 −1——与原代码逐位相同）、起点取 `start.unwrap_or(0.0)`。前提是 Models 已推送，runner 才能以 git 依赖引用它 |

以后会用到、但不挡第 1 版的：rui-nlp 的热启动（`.nl` 的 `x` / `d` 段、参数化重解）、L-BFGS（只有一阶导数的问题）。

## 8. 分期

每一期都有可以客观判定的验收条件。M0 之后，M1 与 M2 / M3 互不依赖，可以并行。

| 期 | 内容 | 验收 | 状态 |
|---|---|---|---|
| **M0 契约与骨架** | `rui-api`、`rui-models`（注册表、调度、explain、会话）、`rui-ref`；`rui-expr`（迁入求值器） | 没有任何私有代码的机器上 `cargo test` 通过；调度表每一行有测试；状态映射不漏项 | 完成。状态映射的"不漏项"由插件里的穷尽 `match` 在编译时保证，M1 的插件落实 |
| **M1 打通建模生态** | `rui-nlp-plugin`、`rui-nl`（补 `V` / `S` / `x` / `d` 段，`.sol` 写出）、`rui-driver`、Dist 仓库 | Pyomo 与 JuMP 各自解 HS071：状态、目标值、原始解、约束对偶、界乘子与同一路径下的 Ipopt 一致；Mittelmann 47 题经 `rui` 驱动的状态与迭代数和现有 runner 一致 | `rui-nl` 读取器已迁入（现有子集，未补段）；其余未开始。先决定界外起点的处理（§5.1） |
| **M2 Python 包** | `rui-python`、`rui.Model`、`rui.minimize`、`compat.cyipopt`、三个平台的 wheel | 干净环境 `pip install` 后：HS071（建模 API）、一个回调问题、Ctrl-C 能中断、Jupyter 里日志正常 | 未开始 |
| **M3 黑箱生态** | `rui-bo-plugin`（含 `AskTell`）、`rui.Optimizer`、Optuna `RuiSampler` | Branin 经 `rui.Model` 自动调度到 bo；Optuna study 含失败试验、`n_jobs > 1` 时不重复提议同一点 | 未开始 |
| **M4 原生接口** | `rui-capi`、`Rui.jl`（MOI 封装）、二进制 `.nl`、AMPL 实测 | JuMP 不经文件解 HS071；AMPL 的 `solve` 走通 | 未开始 |

## 9. 需要你决定的事

1. **开闭源边界。** ~~求解器是否保持闭源、以二进制分发，Models 开源？~~ **已确认（2026-11-19）：是。**
   仍待定：Models 用什么许可证（Apache-2.0 / MIT），二进制的分发条款是什么。在此之前 workspace 设了 `publish = false`，仓库里没有 LICENSE 文件。
2. ~~是否同意公开 `.nl` 读取器和自动微分求值器？~~ **已同意（2026-11-19），已迁入**：`rui-expr`（求值器）、`rui-nl`（读取器）。
   私有 runner 改为依赖它们是 NLP 仓库里的后续一步（§7）。
3. **"Optmodel" 的含义。** 我按 SAS `PROC OPTMODEL` 理解，并把"建模"落在 Python API 上。如果你要的是一门独立的文本建模语言，那是另一个量级的项目，需要单独设计。
4. **名字。** 可执行文件与 Python 包名暂定 `rui`（AMPL 的选项变量随之为 `rui_options`）。PyPI 上是否可用尚未查。
5. **先做哪个入口。** 我的建议是 M1（`.nl` 驱动）先于 M2：代码量最小，覆盖面最大，而且它的验收标准（与 Ipopt 逐项对比）你已经有现成的工具。

## 10. 核实记录

**已核实**

| 结论 | 依据 |
|---|---|
| 两个求解器的接口、状态、选项、限制（§2、§7） | 读了两个仓库的源码与文档：`model/mod.rs`、`ipm/result.rs`、`ipm/options.rs`、`solver.rs`、`space.rs`、`ampl-nlp-runner/src/{nl,eval}.rs` |
| 可选的私有 git 依赖会让公开仓库无法构建 | 本机实验（cargo 1.100.0）：可选 git 依赖指向够不着的仓库，feature 关闭；没有锁文件时构建失败；先在够得着时生成 `Cargo.lock`，再移走仓库并清掉缓存，`cargo build --locked` 仍然失败 |
| M0 的代码按本文 §4 工作 | rustc / cargo 1.100.0（`x86_64-pc-windows-gnu`）：`cargo test --workspace` 70 个测试全部通过（调试与 release 构建、`--offline` 各一次）；`cargo clippy --workspace --all-targets -- -D warnings` 无告警；`cargo fmt --check` 通过；`RUSTDOCFLAGS=-D warnings cargo doc --workspace --no-deps` 通过；`Cargo.lock` 里只有五个 workspace crate，没有外部依赖；`cargo run -p rui-ref --example quickstart` 的输出见 §4.4 |
| 迁入的 `.nl` 读取器与求值器与私有原代码逐位一致 | 一个临时比较程序（不在任何仓库里）同时链接私有的 `ampl-nlp-runner` 与公开的 `rui-nl` + `rui-expr`，读 Mittelmann 的 47 个 `.nl` 文件：变量与约束的界、起点、Jacobian 与 Hessian 结构、规模统计完全相同；在三个点（起点投影到界内、两个扰动）上目标、梯度、约束、Jacobian、Lagrange Hessian 的每个值 `to_bits()` 相同（最大化的 4 题按 σ 换算后比较）。47 题全部一致，release 构建 |
| 分类在真实模型上合理 | 同一批文件经 `Signature` 的类别：`qcqp*` 9 题与 `cont5_2_2/3/4_l`、`ex4_2_*`、`marine`、`nql180`、`optmass`、`pinene`、`qssp180`、`camshape` 共 20 题为 QCQP，`bearing_400`、`cont5_2_1_l` 为 QP，其余 25 题为 NLP；4 题的起点在界外（§5.1） |
| Pyomo 通用 ASL 插件：`options.solver` 是可执行文件名且不传给求解器；命令行 `[可执行文件, 问题文件, '-AMPL']` 加 `key=value`；环境变量 `<solver>_options`；只接受 `.nl`，读回 `.sol`；用 `-v`（超时 5 秒）取版本并据此判断是否可用 | [Pyomo `ASL.py` 源码](https://pyomo.readthedocs.io/en/latest/_modules/pyomo/solvers/plugins/solvers/ASL.html) |
| Pyomo NL writer：只写文本格式（`g3`）；`export_defined_variables` 默认开、写 `V` 段；写 `S`、`x`、`d`、`r`（类型 0–5）、`b`、`k`、`J`、`G`、`F` 段；可写多个目标 | [Pyomo `nl_writer.py` 源码](https://pyomo.readthedocs.io/en/latest/_modules/pyomo/repn/plugins/nl_writer.html) |
| AmplNLWriter：`solver model.nl -AMPL 选项…`，选项只走命令行；读回 `model.sol` | [AmplNLWriter.jl 源码](https://raw.githubusercontent.com/jump-dev/AmplNLWriter.jl/master/src/AmplNLWriter.jl) |
| `solve_result_num` 各区间到 MOI 状态的映射；读取约束对偶与 `ipopt_zL_out` / `lower_bound_duals` 等后缀 | [MathOptInterface `FileFormats/NL/sol.jl` 源码](https://raw.githubusercontent.com/jump-dev/MathOptInterface.jl/master/src/FileFormats/NL/sol.jl) |
| Optuna `BaseSampler` 的四个方法及其调用时机 | [Optuna 文档](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.BaseSampler.html) |
| `cyipopt.Problem` 的回调形状；`minimize_ipopt` 与 `scipy.optimize.minimize` 同形 | [cyipopt 文档](https://cyipopt.readthedocs.io/en/latest/tutorial.html) |

**只有间接依据**

- Pyomo 对未注册的求解器名退回 ASL 插件并设置 `solver=<名字>`：来自搜索结果对 [`pyomo/opt/base/solvers.py`](https://pyomo.readthedocs.io/en/latest/_modules/pyomo/opt/base/solvers.html) 的摘录，没有通读全文。M1 的验收会直接验证。
- OPTMODEL 的 `SOLVE [WITH 求解器] [/ 选项]`、不写 `WITH` 时按问题类型自动选择、写了 `WITH` 才能给选项、LP / MILP / QP / NLP 的对应表、存在黑箱求解器：
  来自搜索结果对 SAS 文档的摘录（[SOLVE 语句](https://support.sas.com/documentation/cdl/en/ormpug/68156/HTML/default/ormpug_optmodel_syntax02.htm)、
  [黑箱求解器](https://documentation.sas.com/doc/en/pgmsascdc/v_077/casmopt/casmopt_lsosolver_overview.htm)）；直接打开 SAS Viya 文档页时返回 404，没有读到原文。
- maturin 支持把 Rust 可执行文件打进 wheel（`bindings = "bin"`）：来自 [maturin 仓库](https://github.com/PyO3/maturin)的说明；"同一个 wheel 里既有扩展模块又有可执行文件"的具体做法没有验证。

**未核实，实现时必须先查**

- AMPL 调用求解器时默认写二进制还是文本 `.nl`。
- `.sol` 文件的确切格式，以及 AMPL / Pyomo / JuMP 各自的对偶符号约定（§4.6 的验收方法就是为此设的）。
- Pyomo 解析 `-v` 输出的具体规则。
- MathOptInterface 的 NL writer 写的是文本格式（由它的 `.sol` 读取方式推测）。
- PyPI 上包名是否可用。
- 支持的最低 Rust 版本：只在 1.100.0 上测试过；代码用到的较新 API（如 `Option::is_none_or`，1.82 起）说明它至少需要 1.82，确切下限没有验证，所以没有在 `Cargo.toml` 里写 `rust-version`。
