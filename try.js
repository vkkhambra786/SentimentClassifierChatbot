// Input: s = "  hello world  "
// Output: "world hello"
// Explanation: Your reversed string should not contain leading or trailing spaces.

// Example 3:

// Input: s = "a good   example"
// Output: "example good a"
// Explanation: You need to reduce multiple spaces between two words to a single space in the reversed string.


var str =  "a good   example" //"  hello world  ";
var result = "";
var word = "";
var inWord = false;

// Go through string from start to end
for (var i = 0; i < str.length; i++) {
    if (str[i] !== ' ') {
        word = word + str[i];
        inWord = true;
    } else {
        if (inWord) {
            // Add word to front of result with space
            if (result.length > 0) {
                result = word + " " + result;
            } else {
                result = word;
            }
            word = "";
            inWord = false;
        }
    }
}

// Handle last word
if (inWord && word.length > 0) {
    if (result.length > 0) {
        result = word + " " + result;
    } else {
        result = word;
    }
}

console.log("Input: '" + str + "'");
console.log("Output: '" + result + "'");