# 🌟 ZenID: Face Swap 🌟
Inspired by [**InstantID**](https://github.com/instantX-research/InstantID) and [**InstantID Comfy**](https://github.com/cubiq/ComfyUI_InstantID)

This **ZenID Node** has been refactored for specialized tasks like _Face Swap_

**✨ Support further development by starring the project! ✨**

## 📑 **Table of Contents** 
1. [Updates](#updates) 
2. [ZenID Features](#zenid-features) 
    * [Face Swap](#zenid-face-swap) 
    * [Face Combine](#zenid-face-combine) 
3. [Installation](#installation)

## 📅 **Updates** <a name="updates"></a>
* **2025/03/27**: 🔥🔥🔥 New Flux Support [InfiniteYou](https://github.com/ZenAI-Comfy/ComfyUI_InfiniteYou) for ZenID 🔥🔥🔥
* **2024/11/19**: Cập nhật README Tiếng Việt [READMEVN](https://github.com/vuongminh1907/ComfyUI_ZenID/blob/main/README_VN.md)
* **2024/11/18**: Released the new **_Face Swap_** functionality.
* **2024/11/16**: Added the **_Face Combine_** feature.


## 🌟 **ZenID Features** <a name="zenid-features"></a>

### 🔗 **ZenID Face Swap** <a name="zenid-face-swap"></a>
- **Workflows**
    Sample [`ZenID_FaceSwap.json`](https://github.com/vuongminh1907/ComfyUI_ZenID/blob/main/workflow/ZenID_FaceSwap.json) are included in the `workflow` folder.
    ![ZenID Face Swap Example](https://github.com/vuongminh1907/ComfyUI_ZenID/blob/main/examples/zenid_faceswap.png)

- **Examples**
    - **Source Images**  
      <img src="https://github.com/vuongminh1907/ComfyUI_ZenID/blob/main/examples/blackwukong.png" width="300" /> 
      <img src="https://github.com/vuongminh1907/ComfyUI_ZenID/blob/main/examples/domixi.png" width="300" />

    - **Result Image**  
       <img src="https://github.com/vuongminh1907/ComfyUI_ZenID/blob/main/examples/result_faceswap.png" width="600" />


### 🔗 **ZenID Face Combine** <a name="zenid-face-swap"></a>
- **Workflows**

    Sample [`ZenID_combineFace.json`](https://github.com/vuongminh1907/ComfyUI_ZenID/blob/main/workflow/ZenID_combineFace.json) are included in the `workflow` folder.
    ![ZenID Face Combine Example](https://github.com/vuongminh1907/ComfyUI_ZenID/blob/main/examples/zenid_combineface.png)
- **Examples**
    - **Source Images**  
      <img src="https://github.com/vuongminh1907/ComfyUI_ZenID/blob/main/examples/haitu.jpg" width="300" /> 
      <img src="https://github.com/vuongminh1907/ComfyUI_ZenID/blob/main/examples/sontung.jpg" width="300" />

    - **Result Image**  
       <img src="https://github.com/vuongminh1907/ComfyUI_ZenID/blob/main/examples/result_facecombine.png" width="600" />

## ⚙️ Installation <a name="installation"></a>
1. Upgrade ComfyUI to the latest version.
```
git clone https://github.com/comfyanonymous/ComfyUI
```
2. Clone or download this repository into the `ComfyUI/custom_nodes/` directory.
```
cd ComfyUI/custom_nodes/
git clone https://github.com/vuongminh1907/ComfyUI_ZenID
```
3. Download the model
```
pip install -r ComfyUI_ZenID/requirements.txt
python ComfyUI_ZenID/downloadmodel.py
```
4. Run ComfyUI 

## Contact for Work 🌟

This is a demo Face Swap product by ZenAI.  
<img src="https://github.com/vuongminh1907/ComfyUI_ZenID/blob/main/examples/zenai.png" width="400" />

If you need more polished and enhanced version, please contact us through:  
- 📱 **Facebook Page**: [ZenAI](https://web.facebook.com/zenai.vn)  
- ☎️ **Phone**: 0971912713 Miss. Chi  

## 📺 Social Network

Check out tutorials, demos, and updates on community's social media channels

### <img src="https://upload.wikimedia.org/wikipedia/commons/4/42/YouTube_icon_%282013-2017%29.png" width="30"> YOUTUBE

1. [ZenID Face Swap｜Generate different ages｜｜ComfyUI｜Workflow Download Installation Setup Tutorial](https://www.youtube.com/watch?v=UnFK-SjkIS0&t=1s)

2. [ZenID Fun & Face Aging Alternative｜Predict Your Child’s Appearance!](https://www.youtube.com/watch?v=d3NMFWHVEiw&t=183s)

3. [The best face swap I have used! Not PuLID! No LoRA Training Required. ComfyUI ZenID](https://www.youtube.com/watch?v=uHU5rtQu4jc&t=1246s)

###  <img src="https://upload.wikimedia.org/wikipedia/vi/thumb/1/1b/Bi%E1%BB%83u_tr%C6%B0ng_Bilibili.svg/239px-Bi%E1%BB%83u_tr%C6%B0ng_Bilibili.svg.png" width="40">  Bilibili


1. [这个插件能搞米！双人照预测孩子长相](https://www.bilibili.com/video/BV1muUtYnEqs/?spm_id_from=333.788.videopod.sections&vd_source=165d5f2dd4fb3d3dec1ffdc609c7f4d6&fbclid=IwZXh0bgNhZW0CMTAAAR1fxwVuECesmANgrEt20nXwhOlItGCbzyGAK3xJU3Gx4TOumqwzzpwdE2E_aem_CdQ3mG_P_X5d78ZnMV1JdQ)

2. [【ComfyUI工作流】当前最新换脸工作流！ZenID Face Swap插件 部署安装使用简单适合新手使用](https://www.bilibili.com/video/BV1CdkuYyEPd/?spm_id_from=333.999.0.0)

3. [【AI摄影应用课程】【Comfyui】comfyui工作流ZenID，基础工作流大改，实现更加便捷快速的换脸，面部融和，参考父母长相预测孩子长相](https://www.bilibili.com/video/BV1fwkhYcEob/?spm_id_from=333.337.search-card.all.click)

4. [ComfyUI ZenID高融合换脸 参考父母长相预测孩子不同年龄段样貌](https://www.bilibili.com/video/BV1Ank3Y7EPU/)

5. [一个简单快速可用的换脸工作流，使用新插件 ZENID，附工作流](https://www.bilibili.com/video/BV1s8kkYrE9F/)

6. [ComfyUI_ZenID：预测孩子长相（面部融合）工作流整合包，还能指定年龄以及调整更偏向哪一方的长相](https://www.bilibili.com/video/BV1ayqZYUECB/)

### <img src="https://cdn.pixabay.com/photo/2021/06/15/12/28/tiktok-6338429_1280.png" width="30">  DOUYIN

1. [设计师学Ai（回归）](https://www.douyin.com/video/7450052025175248162)

2. [Danny](https://www.douyin.com/video/7448537507790114060)

3. [AI—绘画师](https://www.douyin.com/video/7450882731820584232)

### <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/b/b9/2023_Facebook_icon.svg/900px-2023_Facebook_icon.svg.png" width="30"> FACEBOOK
1. [Stable Diffusion Viet Nam](https://www.facebook.com/groups/2402371183273100/posts/swap-face-ch%E1%BB%89-v%E1%BB%9Bi-m%E1%BB%99t-node-l%E1%BA%A5y-c%E1%BA%A3m-h%E1%BB%A9ng-t%E1%BB%AB-instantid-zenid-%C4%91%C6%B0%E1%BB%A3c-t%E1%BA%A1o-ra-ch%E1%BB%89-v%E1%BB%9Bi-v/2866596150183932/?_rdr)

2. [Bình Dân Học AI](https://www.facebook.com/groups/binhdanhocai/permalink/593754026446689/)

### 🌐 Workflow Platforms
1. [Openart.ai](https://openart.ai/workflows/t8star/zenid/zLFXFt5JTgvxjY6IcYwH)

2. [runninghub.ai](https://www.runninghub.ai/post/1868731191779033089)

3. [Liblib.art](https://www.liblib.art/modelinfo/c5fd923aa45b43a5a20ed3cb08c2e081?versionUuid=d327441153884f249569ea90bd08801b)

### ✍️ Blogs

1. [[ComfyUI]ZenID：面部融合神器！揭秘基因遗传，提前预测你和她的下一代基因](https://blog.csdn.net/xiaoganbuaiuk/article/details/144468312)

2. [ComfyUI 面部迁移，面部融合，预测宝宝长相，ZenID 插件安装使用](https://blog.csdn.net/haikun/article/details/144566495)

3. [[ComfyUI Tutorial] What will the baby of the future look like? Double photo reveals the genetic code! Face fusion tool!](https://blog.csdn.net/m0_56144365/article/details/144581826)

🎉 Enjoy creating with ZenID! Let us know your feedback or suggestions.


## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=vuongminh1907/ComfyUI_ZenID&type=Date)](https://star-history.com/#vuongminh1907/ComfyUI_ZenID&Date)
