# Phase 01 验收

17 个场景共 1020 个 bag，11900 条图像出现记录、818 个按场景去重的帧。文件夹只含 images/intrins/poses。构建时核对了跨 bag 同名帧的 RGB 哈希、相机矩阵及分辨率。同场景同 bag 同现、按几何及固定哈希生成，形成 3×100 开发和 14×100 封存测试对。独立 MegaDepth 开发为 8×32 对。详情及逐文件哈希见 manifests/dev_manifest_hashes.json 和 protocol_frozen.md。

两次 prepare_texturepoor.sh 退出 0，输出哈希一致。三项 CLI 单元测试全部通过；实际解码全部 300 个专项开发对两侧图像成功。原始标准输出和退出码位于 logs/phase01。开发/最终模型成绩均未生成。原版 GPU 验证进入 Phase 02。恢复入口为 bash experiments/lowtex_e2e_v1/scripts/prepare_texturepoor.sh。
