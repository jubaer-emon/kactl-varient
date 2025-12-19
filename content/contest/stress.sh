set -e
g++ code.cpp -o code
g++ gen.cpp -o gen
g++ brute.cpp -o brute
for((i = 1; ; ++i)); do
    ./gen $i > in
    ./code < in > out
    ./brute < in > ans
    diff -Z out ans > /dev/null || break
    # diff -w <(./A < int) <(./B < int) || break
    echo "Passed test: "  $i
done
echo "WA on the following test:"
cat in
echo "Your answer is:"
cat out
echo "Correct answer is:"
cat ans