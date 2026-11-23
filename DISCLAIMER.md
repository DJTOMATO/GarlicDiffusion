# Disclaimer, Hardware Safety & Limitation of Liability

Please read this carefully before installing or running **GarlicDiffusion** on your PlayStation 4 console.

---

## 1. No Warranty & Exclusion of Liability

GarlicDiffusion is free, open-source software distributed under the **[GNU General Public License v3.0 (GPLv3)](LICENSE)** on an **"AS IS"** basis, **WITHOUT WARRANTY OF ANY KIND**, either express or implied, including without limitation any implied warranties of merchantability, fitness for a particular purpose, or non-infringement.

In accordance with **GPLv3 Sections 15 and 16**, in no event shall the authors, maintainers, contributors, or upstream copyright holders (including the authors of Garlic LLM, OpenOrbis, or any third-party AI model) be liable for **any direct, indirect, incidental, special, exemplary, or consequential damages** arising out of the use or inability to use this software — including, without limitation:

- Console overheating, thermal shutdown, fan wear, solder/APU degradation, or hardware failure
- Loss of data, corrupted storage, or system instability
- Voided warranties or PlayStation Network account restrictions resulting from running homebrew/jailbreak environments

**You install and run GarlicDiffusion entirely at your own risk.**

---

## 2. Thermal & Aging Hardware Warning (Important!)

PlayStation 4, PS4 Slim, and PS4 Pro consoles are **7 to 13+ years old**. Many consoles still have dried factory thermal paste, clogged heatsinks, or dusty power supplies.

Unlike games whose GPU load fluctuates between frames and loading screens, **neural network diffusion inference drives sustained, dense ALU and GDDR5 memory traffic across Compute Units**:

- **Use `Quiet` (20 CU on Pro / 11 CU on Base) or `Silent` (14 CU on Pro / 7 CU on Base)** for everyday use, unattended batches, or older consoles. Quiet is the default on PS4 Pro for a reason!
- **Avoid `Turbo` (36 CU / 100%)** unless your console is well-ventilated, recently cleaned, and repasted. Turbo draws peak APU power and spins the blower fan aggressively.
- **Ensure clear airflow** around the rear exhaust and side intakes — never run GarlicDiffusion inside a closed cabinet or on carpet.
- **Stop immediately** if your PS4 reports an overheating warning or exhibits abnormal shutdowns, and service the console's dust/thermal paste before resuming heavy compute workloads.
- Note: On userland PS4 homebrew sandboxes, the hardware thermal sensor syscall is restricted by the OS, so GarlicDiffusion controls heat proactively via **symmetric hardware CU masking (`SH_STATIC_THREAD_MGMT`)** and **inter-block thermal cooldown breathers**.

---

## 3. Homebrew & Unofficial Software

- GarlicDiffusion is an independent, unofficial homebrew application built exclusively with the open-source [OpenOrbis PS4 Toolchain](https://github.com/OpenOrbis/OpenOrbis-PS4-Toolchain).
- It contains **no kernel exploit**, no Sony proprietary SDK code, and no copyrighted game assets.
- Running it requires a jailbroken PS4 running [GoldHEN](https://github.com/GoldHEN/GoldHEN), which you install and operate at your own discretion and risk.
- GarlicDiffusion is **not affiliated with, sponsored by, or endorsed by Sony Interactive Entertainment**.

---

## 4. AI Models & Generated Content Responsibility

- GarlicDiffusion is a local, offline execution engine. No prompts or images are transmitted to any external server.
- Model checkpoints are third-party works downloaded separately by the user. Users are solely responsible for complying with the license terms of every model they download or convert, and for ensuring that all generated content complies with applicable local laws and ethical standards.
- The maintainers of GarlicDiffusion assume zero responsibility for user-generated images or third-party model outputs.
