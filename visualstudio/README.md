﻿﻿﻿﻿﻿﻿﻿﻿﻿﻿﻿﻿﻿﻿﻿﻿﻿Greeting from andre :-)

It took me almost 3 weeks to make face mesh build on Visual Studio before I can test and debug easily. So I'd like to share this to anyone who is surfering bazel build and trying to port to Visual Studio like I did. Would be glad if this saves your precious time. Many thanks to mediapipe team for sharing this fantastic work.

Before open up Visual Studio 2019 and make the build, install PowerShell if you haven't yet. Have fun!.

**Now you know everything I know.**

#### History
 * 2022/04/17 init commit.
 * 2022/05/08 add **x86**(32-bit) target platforms, but with few problems... (always prefer **x64** versions)
 
   [Win32 | Debug] Application crashes when ends. Move to other opencv liibrary may fix this (cv::VideoCapture)
   
   [Win32 | Release] A suspicious Visual Studio 2019 ver.16.11.13 bug when linking **tensorflow-static** library. Set **tensorflow-lite** optimization level to **Disabled(/Od)** can work around this. (Set **C/C++ | Optimization** back to **/O2** to experience this **fatal error C1001: Internal compiler error.**)





















