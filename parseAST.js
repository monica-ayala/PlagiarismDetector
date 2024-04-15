const AbstractSyntaxTree = require('abstract-syntax-tree');

// Asynchronously get stdin using ES Modules dynamic import
async function getStdin() {
    const module = await import('get-stdin');
    return module.default();
}

(async () => {
    try {
        // Get input from standard input (stdin)
        const code = await getStdin();

        // Create the AST from the provided JavaScript code
        const ast = new AbstractSyntaxTree(code);

        // Print the AST as a JSON string to standard output
        console.log(JSON.stringify(ast, null, 2));
    } catch (error) {
        console.error('Error parsing input:', error.message);
        process.exit(1);
    }
})();

