((nil . ((eval . (dap-register-debug-template
                  "Python :: fork :: Debug Run"
                  (list :name "Python :: fork :: Debug Run"
                        :type "python"
                        :cwd "${workspaceFolder}"
                        :module nil
                        :program "${workspaceFolder}/main.py"
                        :args '("test" "--plan" "regular" "regular" "fork" "regular" "regular" "fork" "regular" "regular" "--validation_interval" "3" "--report_interval" "3")
                        :request "launch"))))))
