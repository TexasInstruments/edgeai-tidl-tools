/*Copyright (c) 2026 Texas Instruments Incorporated

All rights reserved not granted herein.

Limited License.  

Texas Instruments Incorporated grants a world-wide, royalty-free, non-exclusive
license under copyrights and patents it now or hereafter owns or controls to 
make, have made, use, import, offer to sell and sell ("Utilize") this software
subject to the terms herein.  With respect to the foregoing patent license,
such license is granted  solely to the extent that any such patent is necessary
to Utilize the software alone.  The patent license shall not apply to any
combinations which include this software, other than combinations with devices
manufactured by or for TI (“TI Devices”).  No hardware patent is licensed
hereunder.

Redistributions must preserve existing copyright notices and reproduce this
license (including the above copyright notice and the disclaimer and
(if applicable) source code license limitations below) in the documentation
and/or other materials provided with the distribution

Redistribution and use in binary form, without modification, are permitted
provided that the following conditions are met:

*	No reverse engineering, decompilation, or disassembly of this software is
    permitted with respect to any software provided in binary form.

*	any redistribution and use are licensed by TI for use only with TI Devices.

*	Nothing shall obligate TI to provide you with source code for the software
    licensed and provided to you in object code.

If software source code is provided to you, modification and redistribution of 
the source code are permitted provided that the following conditions are met:

*	any redistribution and use of the source code, including any resulting
    derivative works, are licensed by TI for use only with TI Devices.

*	any redistribution and use of any object code compiled from the source code
    and any resulting derivative works, are licensed by TI for use only with TI 
    Devices.

Neither the name of Texas Instruments Incorporated nor the names of its
suppliers may be used to endorse or promote products derived from this software
without specific prior written permission.

DISCLAIMER.

THIS SOFTWARE IS PROVIDED BY TI AND TI’S LICENSORS "AS IS" AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
EVENT SHALL TI AND TI’S LICENSORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#ifndef ARGUMENTPARSER_H
#define ARGUMENTPARSER_H

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <algorithm>

class ArgParser {
public:
    // Represents a single argument definition
    struct ArgDefinition {
        std::string longName;
        char shortName; // Use 0 for no short name
        bool isFlag;    // True if it's a flag (no value), false if it takes a value
        bool multiValue; // True if it accepts multiple values
        std::string description;
        std::string defaultValue; // For value-taking args
    };

    ArgParser() = default;

    // Add an argument that takes a value
    void addArgument(char shortName, const std::string& longName, const std::string& description, const std::string& defaultValue = "", bool multiValue = false) {
        argDefinitions.push_back({longName, shortName, false, multiValue, description, defaultValue});
    }

    // Add a flag argument (no value)
    void addFlag(char shortName, const std::string& longName, const std::string& description) {
        argDefinitions.push_back({longName, shortName, true, false, description, ""});
    }

    // Parse the command-line arguments
    bool parse(int argc, char* argv[]) {
        // Check for help argument first
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "-h" || arg == "--help" || arg == "-help") {
                printHelp();
                return false; // Return false to indicate that normal execution should stop
            }
        }
        
        // Initialize parsed arguments with defaults
        for (const auto& def : argDefinitions) {
            if (def.isFlag) {
                parsedArgs[def.longName] = "false"; // Flags default to false
            } else {
                parsedArgs[def.longName] = def.defaultValue;
            }
            // Initialize multi-value arguments with empty vectors
            if (def.multiValue) {
                parsedMultiArgs[def.longName] = std::vector<std::string>();
                if (!def.defaultValue.empty()) {
                    parsedMultiArgs[def.longName].push_back(def.defaultValue);
                }
            }
        }

        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];

            // Handle long arguments (e.g., --verbose, --output=file.txt)
            if (arg.rfind("--", 0) == 0 || (arg.rfind("-", 0) == 0 && arg.length() > 1)) {
                std::string name;
                std::string value;
                size_t eqPos = arg.find('=');
                bool isLongFormat = arg.rfind("--", 0) == 0;
                
                if (isLongFormat) {
                    // Long format (--name or --name=value)
                    if (eqPos != std::string::npos) {
                        name = arg.substr(2, eqPos - 2);
                        value = arg.substr(eqPos + 1);
                    } else {
                        name = arg.substr(2);
                    }
                } else {
                    // Short format (-n or -nvalue)
                    char shortNameChar = arg[1];
                    auto it = std::find_if(argDefinitions.begin(), argDefinitions.end(),
                                        [&](const ArgDefinition& def) { return def.shortName == shortNameChar; });
                    
                    if (it != argDefinitions.end()) {
                        name = it->longName;
                        if (arg.length() > 2) { // Value directly attached (e.g., -ofile.txt)
                            value = arg.substr(2);
                            eqPos = 2; // Mark as having a value
                        }
                    } else {
                        std::cerr << "Error: Unknown argument -" << shortNameChar << std::endl;
                        return false;
                    }
                }

                // Find the argument definition
                auto it = std::find_if(argDefinitions.begin(), argDefinitions.end(),
                                    [&](const ArgDefinition& def) { return def.longName == name; });

                if (it != argDefinitions.end()) {
                    if (it->isFlag) {
                        parsedArgs[it->longName] = "true";
                    } else {
                        if (eqPos != std::string::npos) {
                            // Value provided with = or attached to short name
                            if (it->multiValue) {
                                parsedMultiArgs[it->longName].push_back(value);
                            } else {
                                parsedArgs[it->longName] = value;
                            }
                        } else if (i + 1 < argc) {
                            if (it->multiValue) {
                                // For multi-value arguments, collect all subsequent values until we hit another flag/option
                                while (i + 1 < argc && argv[i+1][0] != '-') {
                                    value = argv[++i];
                                    parsedMultiArgs[it->longName].push_back(value);
                                }
                                // If no values were collected, that's okay for multi-value args
                            } else if (argv[i+1][0] != '-') {
                                // For single-value args, just take the next argument if it's not a flag
                                value = argv[++i];
                                parsedArgs[it->longName] = value;
                            } else if (it->multiValue) {
                                // For multi-value args, it's okay to have no value (will use default)
                            } else {
                                std::cerr << "Error: Argument " << (isLongFormat ? "--" : "-") << name << " requires a value." << std::endl;
                                return false;
                            }
                        } else if (it->multiValue) {
                            // For multi-value args, it's okay to have no value (will use default)
                        } else {
                            std::cerr << "Error: Argument " << (isLongFormat ? "--" : "-") << name << " requires a value." << std::endl;
                            return false;
                        }
                    }
                } else {
                    std::cerr << "Error: Unknown argument " << arg << std::endl;
                    return false;
                }
            }
            else {
                // Handle positional arguments if needed, or report unknown
                std::cerr << "Error: Unknown argument: " << arg << std::endl;
                return false;
            }
        }
        return true;
    }

    // Get the value of an argument
    std::string getValue(const std::string& longName) const {
        auto it = parsedArgs.find(longName);
        if (it != parsedArgs.end()) {
            return it->second;
        }
        return ""; // Or throw an exception for unknown argument
    }

    // Get multiple values for an argument
    std::vector<std::string> getMultiValues(const std::string& longName) const {
        auto it = parsedMultiArgs.find(longName);
        if (it != parsedMultiArgs.end()) {
            return it->second;
        }
        return std::vector<std::string>(); // Return empty vector if not found
    }

    // Check if a flag is set
    bool getFlag(const std::string& longName) const {
        return getValue(longName) == "true";
    }

    // Print help message
    void printHelp() const {
        std::cout << "Usage: program_name [options]" << std::endl;
        std::cout << "Options:" << std::endl;
        for (const auto& def : argDefinitions) {
            std::cout << "  ";
            if (def.shortName != 0) {
                std::cout << "-" << def.shortName << ", ";
            }
            std::cout << "--" << def.longName;
            if (!def.isFlag) {
                std::cout << " <value>";
            }
            std::cout << "  " << def.description;
            if (!def.isFlag && !def.defaultValue.empty()) {
                std::cout << " (Default: " << def.defaultValue << ")";
            }
            std::cout << std::endl;
        }
    }

private:
    std::vector<ArgDefinition> argDefinitions;
    std::map<std::string, std::string> parsedArgs;
    std::map<std::string, std::vector<std::string>> parsedMultiArgs;
};

#endif // ARGUMENTPARSER_H
