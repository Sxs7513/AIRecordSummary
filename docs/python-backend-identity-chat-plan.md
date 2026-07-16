# Python 后端：账号与 Workspace 授权方案

## 1. 目标

在 Python 后端建立账号、Session 与 Workspace 授权边界，使录音、转写、总结、RAG 检索和音频读取均只能由有权用户访问。

本阶段原则：

- Workspace 是录音的默认授权边界，不要求对同一团队的每条录音逐条授权。
- `recording_memberships` 仅用于跨 Workspace 或例外的单条录音分享。
- RAG 的授权必须在数据库检索层强制执行，不能依赖前端传参或模型判断。
- 多轮聊天、聊天消息与前端消息状态设计见 [多轮录音问答与前端聊天方案](python-backend-multi-turn-chat-plan.md)。

## 2. 身份认证

第一版由 Python API 负责账号密码与服务端 Session。

### 2.1 数据表

```text
users
  ├── id
  ├── email                 唯一登录标识
  ├── display_name
  ├── password_hash
  ├── current_workspace_id  当前默认 Workspace；第一版仅由数据库维护
  ├── status                active / disabled
  ├── created_at
  └── updated_at

user_sessions
  ├── id
  ├── user_id
  ├── token_hash            只存 Cookie 随机值的哈希
  ├── expires_at
  ├── revoked_at
  ├── last_seen_at
  ├── created_at
  └── updated_at
```

登录成功后，Python API 设置 `HttpOnly`、`SameSite=Lax` 的 Session Cookie。生产环境增加 `Secure`；服务端只保存 token hash，以支持登出、撤销与设备管理。

首版不使用 JWT。当前单体部署下，数据库 Session 的撤销语义更明确；未来接入 OAuth 时可增加 `auth_identities`，不改变业务授权模型。

### 2.2 本地初始化

`npm run db:init` 重建基线表后，读取根目录 `.env` 的 `BOOTSTRAP_ADMIN_EMAIL`、`BOOTSTRAP_ADMIN_PASSWORD` 和 `BOOTSTRAP_WORKSPACE_NAME`，创建首个 owner 用户、默认 Workspace 与 membership。默认值仅适合本地开发；开始使用前应在 `.env` 覆盖管理员密码。

首版没有注册页面、成员管理页或 Workspace 切换器；需要变更用户默认 Workspace 时直接维护 `users.current_workspace_id`，但该值必须已经存在于 `workspace_memberships`。

### 2.3 统一身份依赖

FastAPI 提供 `require_current_user`：

1. 从 Cookie 读取 Session token；
2. 计算 hash 并查询未过期、未撤销的 `user_sessions`；
3. 加载有效 `users` 记录；
4. 返回类型化 `CurrentUser`。

所有录音、聊天、Generation SSE 与音频下载接口均依赖该对象，不接受客户端提交的 `user_id`。

浏览器请求 Python API 使用 `credentials: "include"`；Python CORS 启用允许凭据。Next Server Component 调 Python API 时需转发当前请求 Cookie。

## 3. Workspace 与录音授权

### 3.1 默认授权模型

```text
users
  └── workspace_memberships ──> workspaces
                                    └── recordings
```

```text
workspaces
  ├── id
  ├── name
  ├── created_at
  └── updated_at

workspace_memberships
  ├── workspace_id
  ├── user_id
  ├── role                  owner / admin / member
  ├── created_at
  └── primary key (workspace_id, user_id)

recordings
  ├── workspace_id          默认授权边界
  └── owner_user_id         创建者与审计归属
```

`users` 与 `workspaces` 是多对多关系：同一用户可以拥有任意多条 `workspace_memberships`，因此可同时属于多个 Workspace。`users.current_workspace_id` 只表示当前默认上下文，必须同时是该用户的一条有效 membership；它不替代多对多授权关系。

用户是录音所属 Workspace 的成员，即可按 Workspace 角色访问该录音。创建录音时，当前用户必须是目标 Workspace 的成员，且写入 `owner_user_id`。

### 3.2 例外单条分享

`recording_memberships` 不承担团队内默认授权，而只表达例外分享：

```text
recording_memberships
  ├── recording_id
  ├── user_id
  ├── role                  viewer / editor
  ├── granted_by_user_id
  ├── created_at
  └── primary key (recording_id, user_id)
```

有效访问条件为：用户属于 `recording.workspace_id`，或拥有 `recording_memberships(recording_id, user_id)`。

### 3.3 授权收口

建立 `RecordingAccessService`，集中提供：

- `require_view(recording_id, current_user)`
- `require_edit(recording_id, current_user)`
- `accessible_recording_scope(current_user)`

录音详情、上传、修改、删除、重试、流水线状态、转写、总结、音频读取，以及通过 `pipeline_run_id` 或 `generation_run_id` 的间接访问，都先回溯到 recording 后执行授权。

音频文件不直接暴露 `storage_path`；客户端改为访问受鉴权保护的 `GET /api/recordings/{id}/audio`。

## 4. 前端账号与默认 Workspace

第一版新增前端账号模块，但**暂不提供 Workspace 切换器**：

```text
app/
  login/page.tsx                 登录页
  account/page.tsx               当前账号与当前 Workspace 信息页
  sdk/auth/
    client.ts                    login / logout / me / workspace API
    store.ts                     Zustand：当前用户、默认 Workspace、初始化状态
    types.ts                     User、Workspace、Membership 类型
components/
  auth-provider.tsx              应用启动时恢复 GET /api/auth/me
  account-menu.tsx               当前用户、登出入口
```

Session Cookie 是 HttpOnly，前端不保存 token，也不在 Zustand 或 localStorage 中保存 Session。`AuthProvider` 在应用启动时请求 `GET /api/auth/me`，将用户、可访问 Workspace 列表和默认 Workspace 写入 Zustand。

登录调用 `POST /api/auth/login`，必须带 `credentials: "include"`；登出调用 `POST /api/auth/logout`，清空本地账号态并跳转 `/login`。

`GET /api/auth/me` 返回当前用户、全部 memberships，以及由 `users.current_workspace_id` 指定的默认 Workspace。第一版录音列表、上传和新建会话都使用这个默认 Workspace；不接受前端自行选择任意 Workspace ID。

当前 Workspace 暂由直接修改数据库的 `users.current_workspace_id` 变更。后端每次读取仍验证用户存在相应 membership；不合法或为空时返回“未配置 Workspace”，前端展示引导状态。未来增加切换器时，只需增加受控的切换接口与组件，不改变授权查询。

前端路由分为：

- 未登录：只允许 `/login`；
- 已登录但未配置有效默认 Workspace：显示“请联系管理员配置工作空间”；
- 已配置默认 Workspace：显示该 Workspace 的录音、会话和成员可见内容。

## 5. 实施顺序

1. 创建 users、sessions、workspaces、workspace_memberships，完成登录、登出与 `CurrentUser`。
2. 为 recordings 增加 `workspace_id`、`owner_user_id`；实现 `RecordingAccessService` 并收口录音、流水线和音频授权。
3. 按 [多轮录音问答与前端聊天方案](python-backend-multi-turn-chat-plan.md) 实现会话、消息、Generation 关联、RAG 历史和聊天 UI。
4. 覆盖越权录音、越权 source、越权 SSE、跨 Workspace 分享、撤销权限后的再查询等测试。
