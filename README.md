# 照片墙

这是一个纯本地单页`index.html`+`assets.js`的照片墙应用，通过`photowall assets`子命令扫描`assets`目录后生成`assets.js`文件，在`index.html`中会加载`assets.js`文件从而展示照片。

通过运行`uv run photowall tags`来生成`tags.json`文件，然后运行`uv run photowall assets`来生成最终的`assets.js`文件，`assets`子命令会参照`tags.json`中的描述来给图片添加标签。

## 快速开始

安装[uv](https://docs.astral.sh/uv/getting-started/installation/)

把图片和视频放进`assets`文件夹，不需要特别分门别类，直接扔进去就行。
> 最开始没有`assets`文件夹需要手动创建。

运行`init.bat`初始化元数据。

打开`index.html`即可。
> 如果更新了`assets`的内容则再次运行`init.bat`即可。

如果想用启动本地服务器也可以直接运行`start.bat`即可，这个会自动启动本地服务器并且使用浏览器打开。
