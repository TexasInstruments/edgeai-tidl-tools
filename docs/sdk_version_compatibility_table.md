# SDK Version Compatibility

This page provides a compatibility table for compatibilty of EdgeAI TIDL Tools with the SDK.

## Table of Contents
- [Compatibility Types](#compatibility-types)
- [Versioning Scheme](#versioning-scheme)
- [SDK Version 11.02.xx.xx](#sdk-version-1102xxxx)
- [SDK Version 11.01.xx.xx](#sdk-version-1101xxxx)
- [SDK Version 11.00.xx.xx](#sdk-version-1100xxxx)
- [SDK Version 10.01.xx.xx](#sdk-version-1001xxxx)
- [SDK Version 10.00.xx.xx](#sdk-version-1000xxxx)
- [SDK Version 09.02.xx.xx](#sdk-version-0902xxxx)
- [SDK Version 09.01.xx.xx](#sdk-version-0901xxxx)
- [SDK Version 09.00.xx.xx](#sdk-version-0900xxxx)
- [SDK Version 08.06.xx.xx](#sdk-version-0806xxxx)
- [SDK Version 08.05.xx.xx](#sdk-version-0805xxxx)

## Compatibility Types

### Default
The standard release for the specified SDK version.

### Patch with default compatibility
These releases include additional features and fixes and are designed to their corresponding default SDK versions.

### Patch with backward compatibility
These releases include additional features and fixes and are designed to work on backward SDK versions. 

> [NOTE]
For patch releases, the `firmware` and `libraries` needs to be updated on the SOC. Please follow the steps in the `update_target.md` document to update.
> - For Version < 11_02_xx_xx, checkout to the particular version and follows the steps in **docs/update_target.md**
> - For Version >= 11_02_xx_xx, checkout to the particular version and follows the steps in the "TARGET Setup Scripts Overview" section of **scripts/setup/README.md**

## Versioning Scheme

In the edgeai-tidl-tools version format `MAJOR_MINOR_PATCH_BUILD`:

- **First two numbers** (e.g., `11_01` in `11_01_07_00`): Indicate the SDK version that the release is built on top of
- **3rd number**:
  - **Even 3rd number** (e.g., `06` in `11_01_06_00`): Usually indicates a default release or a patch on default
  - **Odd 3rd number** (e.g., `07` in `11_01_07_00`): Usually indicates a backward compatible release


## SDK Version 11.02.xx.xx

- **`11_02_04_00`** - [Default](#default)

> These releases have been validated on:
> - AM62A - N/A
> - J722S \| TDA4AEN \| AM67A - N/A
> - J721E \| TDA4VM           - PSDK LINUX 11.02.00.04 / PSDK RTOS 11.02.00.06
> - J721S2 \| TDA4VL \| AM68A - PSDK LINUX 11.02.00.04 / PSDK RTOS 11.02.00.06
> - J784S4 \| TDA4VH \| AM69A - PSDK LINUX 11.02.00.04 / PSDK RTOS 11.02.00.06
> - AM62 - N/A

## SDK Version 11.01.xx.xx

- **`11_01_06_00`** - [Default](#default)

> These releases have been validated on:
> - AM62A - PSDK LINUX 11.01.07.05
> - J722S \| TDA4AEN \| AM67A - PSDK LINUX 11.01.00.03 / PSDK RTOS 11.01.00.04
> - J721E \| TDA4VM           - PSDK LINUX 11.01.00.03 / PSDK RTOS 11.01.00.04 
> - J721S2 \| TDA4VL \| AM68A - PSDK LINUX 11.01.00.03 / PSDK RTOS 11.01.00.04 
> - J784S4 \| TDA4VH \| AM69A - PSDK LINUX 11.01.00.03 / PSDK RTOS 11.01.00.04 
> - AM62 - PSDK LINUX 11.01.05.03

## SDK Version 11.00.xx.xx

- **`11_01_07_00`** - [Patch with backward compatibility](#patch-with-backward-compatibility)
- **`11_01_05_00`** - [Patch with backward compatibility](#patch-with-backward-compatibility)
- **`11_00_08_00`** - [Patch with default compatibility](#patch-with-default-compatibility)
- **`11_00_06_00`** - [Default](#default)

> These releases have been validated on:
> - AM62A - N/A
> - J722S \| TDA4AEN \| AM67A - PSDK LINUX 11.00.00.08 / PSDK RTOS 11.00.00.06
> - J721E \| TDA4VM           - PSDK LINUX 11.00.00.08 / PSDK RTOS 11.00.00.06
> - J721S2 \| TDA4VL \| AM68A - PSDK LINUX 11.00.00.08 / PSDK RTOS 11.00.00.06
> - J784S4 \| TDA4VH \| AM69A - PSDK LINUX 11.00.00.08 / PSDK RTOS 11.00.00.06
> - AM62 - N/A

## SDK Version 10.01.xx.xx

- **`11_00_07_00`** - [Patch with backward compatibility](#patch-with-backward-compatibility)
- **`10_01_04_00`** - [Patch with default compatibility](#patch-with-default-compatibility)
- **`10_01_00_02`** - [Default](#default)

> These releases have been validated on:
> - AM62A - PSDK LINUX 10.01.00.05
> - J722S \| TDA4AEN \| AM67A - PSDK LINUX 10.01.00.04 / PSDK RTOS 10.01.00.04
> - J721E \| TDA4VM           - PSDK LINUX 10.01.00.04 / PSDK RTOS 10.01.00.04
> - J721S2 \| TDA4VL \| AM68A - PSDK LINUX 10.01.00.04 / PSDK RTOS 10.01.00.04
> - J784S4 \| TDA4VH \| AM69A - PSDK LINUX 10.01.00.05 / PSDK RTOS 10.01.00.04
> - AM62 - PSDK LINUX 10.01.00.05 / 10.01.10.04

## SDK Version 10.00.xx.xx

- **`10_01_03_00`** - [Patch with backward compatibility](#patch-with-backward-compatibility)
- **`10_00_08_00`** - [Patch with default compatibility](#patch-with-default-compatibility)
- **`10_00_06_00`** - [Patch with default compatibility](#patch-with-default-compatibility)
- **`10_00_04_00`** - [Patch with default compatibility](#patch-with-default-compatibility)
- **`10_00_02_00`** - [Default](#default)

> These releases have been validated on:
> - AM62A - PSDK LINUX 10.00.00.08
> - J722S \| TDA4AEN \| AM67A - PSDK LINUX 10.00.00.08 / PSDK RTOS 10.00.00.05
> - J721E \| TDA4VM           - PSDK LINUX 10.00.00.08 / PSDK RTOS 10.00.00.05
> - J721S2 \| TDA4VL \| AM68A - PSDK LINUX 10.00.00.08 / PSDK RTOS 10.00.00.05
> - J784S4 \| TDA4VH \| AM69A - PSDK LINUX 10.00.00.08 / PSDK RTOS 10.00.00.05
> - AM62 - PSDK LINUX 10.00.07.04

## SDK Version 09.02.xx.xx
- **`10_00_07_00`** - [Patch with backward compatibility](#patch-with-backward-compatibility)
- **`10_00_05_00`** - [Patch with backward compatibility](#patch-with-backward-compatibility)
- **`10_00_03_00`** - [Patch with backward compatibility](#patch-with-backward-compatibility)
- **`09_02_09_00`** - [Patch with default compatibility](#patch-with-default-compatibility)
- **`09_02_07_00`** - [Patch with default compatibility](#patch-with-default-compatibility)
- **`09_02_06_00`** - [Default](#default)

> These releases have been validated on:
> - AM62A - PSDK LINUX 09.02.00.05
> - J722S \| TDA4AEN \| AM67A - PSDK LINUX 09.02.00.05 / PSDK RTOS 09.02.00.05
> - J721E \| TDA4VM           - PSDK LINUX 09.02.00.05 / PSDK RTOS 09.02.00.05
> - J721S2 \| TDA4VL \| AM68A - PSDK LINUX 09.02.00.05 / PSDK RTOS 09.02.00.05
> - J784S4 \| TDA4VH \| AM69A - PSDK LINUX 09.02.00.05 / PSDK RTOS 09.02.00.05
> - AM62 - PSDK LINUX 09.02.01.09

## SDK Version 09.01.xx.xx

- **`09_01_07_00`** - [Patch with default compatibility](#patch-with-default-compatibility)
- **`09_01_06_00`** - [Patch with default compatibility](#patch-with-default-compatibility)
- **`09_01_04_00`** - [Patch with default compatibility](#patch-with-default-compatibility)
- **`09_01_03_00`** - [Patch with default compatibility](#patch-with-default-compatibility)
- **`09_01_01_01`** - [Patch with default compatibility](#patch-with-default-compatibility)
- **`09_01_00_05`** - [Patch with default compatibility](#patch-with-default-compatibility)
- **`09_01_00_02`** - [Default](#default)

> These releases have been validated on:
> - AM62A - PSDK LINUX 09.01.00.07
> - J721E \| TDA4VM           - PSDK LINUX 09.01.00.06 / PSDK RTOS 09.01.00.06
> - J721S2 \| TDA4VL \| AM68A - PSDK LINUX 09.01.00.06 / PSDK RTOS 09.01.00.06
> - J784S4 \| TDA4VH \| AM69A - PSDK LINUX 09.01.00.06 / PSDK RTOS 09.01.00.06
> - AM62 - PSDK LINUX 09.01.00.08

## SDK Version 09.00.xx.xx

- **`09_00_00_07`** - [Patch with default compatibility](#patch-with-default-compatibility)
- **`09_00_00_06`** - [Default](#default)

> These releases have been validated on:
> - AM62A - PSDK LINUX 09.00.00.08
> - J721E \| TDA4VM           - PSDK LINUX 09.00.00.08 / PSDK RTOS 09.00.00.02
> - J721S2 \| TDA4VL \| AM68A - PSDK LINUX 09.00.00.08 / PSDK RTOS 09.00.00.02
> - J784S4 \| TDA4VH \| AM69A - PSDK LINUX 09.00.00.08 / PSDK RTOS 09.00.00.02
> - AM62 - PSDK LINUX 08.06.00.02

## SDK Version 08.06.xx.xx

- **`08_06_00_03`** - [Patch with default compatibility](#patch-with-default-compatibility)
- **`08_06_00_02`** - [Default](#default)

> These releases have been validated on:
> - AM62A - PSDK LINUX 08.06.00.45 / FIRMWARE-BUILDER 08.06.00.41
> - J721E \| TDA4VM           - PSDK LINUX 08.06.00.11 / PSDK RTOS 08.06.00.12
> - J721S2 \| TDA4VL \| AM68A - PSDK LINUX 08.06.00.10 / PSDK RTOS 08.06.00.11
> - J784S4 \| TDA4VH \| AM69A - PSDK LINUX 08.06.00.12 / PSDK RTOS 08.06.00.14
> - AM62 - PSDK LINUX 08.06.00.02

## SDK Version 08.05.xx.xx

- **`08_05_00_11`** - [Default](#default)

> These releases have been validated on:
> - J721E \| TDA4VM           - PSDK LINUX 08.05.00.11
> - AM62 - PSDK LINUX 08.05.00.11
