
# EdgeAI TIDL Tools Tags for compatible SDK version

## Release version convention:
- Processor SDK RTOS follows versioning scheme as Major.Minor.Patch.Build. Processor SDK RTOS provides TIDL firmware as part of its release
- Edge AI TIDL tools (this repository) provides release of tools for model compilation and host emulation inference It also follows the same versioning convention.
- Releases having same major number belong to same “release family”. Releases having same pair of {major.minor} version are referred as “release line” here
- The tools are always maintained compatible to a firmware version of same “release line”. For example tools version of 9.2.7.0 is compatible with Processor SDK RTOS 9.2.5.0. Detailed version compatibility is provided in table

---

<div align="center">

|EdgeAI TIDL Tools TAG         |           AM62 |           AM62A |          AM68A/J721S2 (TDA4AL, TDA4VL) |          AM68PA/J721E (TDA4VM)|          AM69A/J784S4(TDA4AP, TDA4VP,TDA4AH, TDA4VH)| AM67A (J722S)|          Notes|
| ---------------------------- |:--------------:|:---------------:|:--------------:|:--------------:|:-------------:|:-------------:|:-------------:|
|  10_00_04_00                  |   10_00_07_04  |     Processor SDK LINUX : 10.00.00.08 | Processor SDK LINUX 10.00.00.08<br /> Processor SDK RTOS 10.00.00.05   |   Processor SDK LINUX 10.00.00.08<br /> Processor SDK RTOS 10.00.00.05  | Processor SDK LINUX 10.00.00.08<br /> Processor SDK RTOS 10.00.00.05   | Processor SDK LINUX 10.00.00.08<br /> Processor SDK RTOS 10.00.00.05   |          |
|  10_00_03_00                  |   09_02_01_09  |     Processor SDK LINUX : 09.02.00.05 | Processor SDK LINUX 09.02.00.05<br /> Processor SDK RTOS 09.02.00.05   |   Processor SDK LINUX 09.02.00.05<br /> Processor SDK RTOS 09.02.00.05  | Processor SDK LINUX 09.02.00.05<br /> Processor SDK RTOS 09.02.00.05   | Processor SDK LINUX 09.02.00.05<br /> Processor SDK RTOS 09.02.00.05   |  This release is backward compatible with SDK 9.2. Please follow the steps [here](backward_compatibility.md) to enable compatibility        |
|  10_00_02_00                  |   10_00_07_04  |     Processor SDK LINUX : 10.00.00.08 | Processor SDK LINUX 10.00.00.08<br /> Processor SDK RTOS 10.00.00.05   |   Processor SDK LINUX 10.00.00.08<br /> Processor SDK RTOS 10.00.00.05  | Processor SDK LINUX 10.00.00.08<br /> Processor SDK RTOS 10.00.00.05   | Processor SDK LINUX 10.00.00.08<br /> Processor SDK RTOS 10.00.00.05   |          |
|  09_02_09_00                  |   09_02_01_09  |     Processor SDK LINUX : 09.02.00.05 | Processor SDK LINUX 09.02.00.05<br /> Processor SDK RTOS 09.02.00.05   |   Processor SDK LINUX 09.02.00.05<br /> Processor SDK RTOS 09.02.00.05  | Processor SDK LINUX 09.02.00.05<br /> Processor SDK RTOS 09.02.00.05   | Processor SDK LINUX 09.02.00.05<br /> Processor SDK RTOS 09.02.00.05   |          |
|  09_02_07_00                  |   09_02_01_09  |     Processor SDK LINUX : 09.02.00.05 | Processor SDK LINUX 09.02.00.05<br /> Processor SDK RTOS 09.02.00.05   |   Processor SDK LINUX 09.02.00.05<br /> Processor SDK RTOS 09.02.00.05  | Processor SDK LINUX 09.02.00.05<br /> Processor SDK RTOS 09.02.00.05   | Processor SDK LINUX 09.02.00.05<br /> Processor SDK RTOS 09.02.00.05   |          |
|  09_02_06_00                  |   09_02_01_09  |     Processor SDK LINUX : 09.02.00.05 | Processor SDK LINUX 09.02.00.05<br /> Processor SDK RTOS 09.02.00.05   |   Processor SDK LINUX 09.02.00.05<br /> Processor SDK RTOS 09.02.00.05  | Processor SDK LINUX 09.02.00.05<br /> Processor SDK RTOS 09.02.00.05   | Processor SDK LINUX 09.02.00.05<br /> Processor SDK RTOS 09.02.00.05   |          |
|  09_01_07_00                  |   09_01_00_08  |     Processor SDK LINUX : 09.01.00.07 | Processor SDK LINUX 09.01.00.06<br /> Processor SDK RTOS 09.01.00.06   |   Processor SDK LINUX 09.01.00.06<br /> Processor SDK RTOS 09.01.00.06  | Processor SDK LINUX 09.01.00.06<br /> Processor SDK RTOS 09.01.00.06   | NA |          |
|  09_01_06_00                  |   09_01_00_08  |     Processor SDK LINUX : 09.01.00.07 | Processor SDK LINUX 09.01.00.06<br /> Processor SDK RTOS 09.01.00.06   |   Processor SDK LINUX 09.01.00.06<br /> Processor SDK RTOS 09.01.00.06  | Processor SDK LINUX 09.01.00.06<br /> Processor SDK RTOS 09.01.00.06   | NA |          |
|  09_01_04_00                  |   09_01_00_08  |     Processor SDK LINUX : 09.01.00.07 | Processor SDK LINUX 09.01.00.06<br /> Processor SDK RTOS 09.01.00.06   |   Processor SDK LINUX 09.01.00.06<br /> Processor SDK RTOS 09.01.00.06  | Processor SDK LINUX 09.01.00.06<br /> Processor SDK RTOS 09.01.00.06   | NA |          |
|  09_01_03_00                  |   09_01_00_08  |     Processor SDK LINUX : 09.01.00.07 | Processor SDK LINUX 09.01.00.06<br /> Processor SDK RTOS 09.01.00.06   |   Processor SDK LINUX 09.01.00.06<br /> Processor SDK RTOS 09.01.00.06  | Processor SDK LINUX 09.01.00.06<br /> Processor SDK RTOS 09.01.00.06   | NA |          |
|  09_01_01_01                  |   09_01_00_08  |     Processor SDK LINUX : 09.01.00.07 | Processor SDK LINUX 09.01.00.06<br /> Processor SDK RTOS 09.01.00.06   |   Processor SDK LINUX 09.01.00.06<br /> Processor SDK RTOS 09.01.00.06  | Processor SDK LINUX 09.01.00.06<br /> Processor SDK RTOS 09.01.00.06   | NA |          |
|  09_01_00_05                  |   09_01_00_08  |     Processor SDK LINUX : 09.01.00.07 | Processor SDK LINUX 09.01.00.06<br /> Processor SDK RTOS 09.01.00.06   |   Processor SDK LINUX 09.01.00.06<br /> Processor SDK RTOS 09.01.00.06  | Processor SDK LINUX 09.01.00.06<br /> Processor SDK RTOS 09.01.00.06   | NA |          |
|  09_01_00_02                  |   09_01_00_08  |     Processor SDK LINUX : 09.01.00.07 | Processor SDK LINUX 09.01.00.06<br /> Processor SDK RTOS 09.01.00.06   |   Processor SDK LINUX 09.01.00.06<br /> Processor SDK RTOS 09.01.00.06  | Processor SDK LINUX 09.01.00.06<br /> Processor SDK RTOS 09.01.00.06   | NA |          |
|  09_00_00_07                  |   08_06_00_02  |    Processor SDK LINUX : 09.00.00.08 | Processor SDK LINUX 09.00.00.08<br /> Processor SDK RTOS 09.00.00.02   |   Processor SDK LINUX 09.00.00.08<br /> Processor SDK RTOS 09.00.00.02  | Processor SDK LINUX 09.00.00.08<br /> Processor SDK RTOS 09.00.00.02   | NA |          |
|  09_00_00_06                  |   08_06_00_02  |    Processor SDK LINUX : 09.00.00.08 | Processor SDK LINUX 09.00.00.08<br /> Processor SDK RTOS 09.00.00.02   |   Processor SDK LINUX 09.00.00.08<br /> Processor SDK RTOS 09.00.00.02  | Processor SDK LINUX 09.00.00.08<br /> Processor SDK RTOS 09.00.00.02   | NA |          |
| 08_06_00_03                  |   08_06_00_02  |    Processor SDK LINUX : 08.06.00.45<br /> FIRMWARE-BUILDER : 08.06.00.41  | Processor SDK LINUX 08_06_00_10<br /> Processor SDK RTOS 08_06_00_11   |   Processor SDK LINUX 08_06_00_11<br /> Processor SDK RTOS 08_06_00_12  | Processor SDK LINUX 08_06_00_12<br /> Processor SDK RTOS 08_06_00_14   | NA |          |
| 08_06_00_02                  |   08_06_00_02  |    Processor SDK LINUX : 08.06.00.45<br /> FIRMWARE-BUILDER : 08.06.00.41  |  08_06_00_38   |   08_06_00_38  | 08_06_00_38   | NA |
| 08_05_00_11                  |   08_05_00_11  |    NA           |            NA  |   08_05_00_11  |           NA  | NA|          |

</div>

---
