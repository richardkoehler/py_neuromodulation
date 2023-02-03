for %%a in ("*.py") do (
    copy "insert_me.txt"+"%%~a" "%%~a.tmp" /B
    move /Y "%%~a.tmp" "%%~a"
)
PAUSE