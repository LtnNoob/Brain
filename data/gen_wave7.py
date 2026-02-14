#!/usr/bin/env python3
"""Generate wave7_cs_math.json - massive CS & Math knowledge base."""
import json

concepts = []
relations = []

def C(label, definition, typ="DEFINITION", trust=0.95):
    concepts.append({"label": label, "definition": definition, "type": typ, "trust": round(trust, 2)})

def R(src, tgt, typ="RELATES_TO", w=0.9):
    relations.append({"source": src, "target": tgt, "type": typ, "weight": round(w, 2)})

# ============================================================
# PROGRAMMING LANGUAGES (70+)
# ============================================================
langs = {
    "Python": "High-level interpreted programming language emphasizing readability and versatility",
    "JavaScript": "Dynamic scripting language for web development, runs in browsers and Node.js",
    "TypeScript": "Typed superset of JavaScript that compiles to plain JavaScript",
    "Java": "Object-oriented compiled language running on the JVM, write once run anywhere",
    "C": "Low-level systems programming language with direct memory access",
    "C++": "Extension of C with object-oriented features, templates, and RAII",
    "C#": "Microsoft's object-oriented language for the .NET platform",
    "Go": "Statically typed compiled language by Google with built-in concurrency",
    "Rust": "Systems language focused on memory safety without garbage collection",
    "Swift": "Apple's modern language for iOS/macOS development",
    "Kotlin": "Modern JVM language, official for Android development",
    "Ruby": "Dynamic object-oriented language emphasizing developer happiness",
    "PHP": "Server-side scripting language widely used for web development",
    "Perl": "High-level text-processing language known for regular expressions",
    "R": "Statistical computing language for data analysis and visualization",
    "MATLAB": "Numerical computing environment for matrix operations and engineering",
    "Julia": "High-performance language for scientific and numerical computing",
    "Scala": "JVM language combining object-oriented and functional programming",
    "Haskell": "Purely functional programming language with strong static typing",
    "Erlang": "Functional language designed for concurrent distributed fault-tolerant systems",
    "Elixir": "Dynamic functional language running on the Erlang VM (BEAM)",
    "Clojure": "Modern Lisp dialect running on the JVM with immutable data structures",
    "F#": "Functional-first language for the .NET platform",
    "OCaml": "Multi-paradigm language with strong type inference from the ML family",
    "Lua": "Lightweight embeddable scripting language often used in game engines",
    "Dart": "Client-optimized language by Google, used with Flutter framework",
    "Objective-C": "Superset of C with Smalltalk-style messaging, legacy Apple development",
    "Assembly Language": "Low-level language with direct CPU instruction mnemonics",
    "COBOL": "Legacy business-oriented language still running banking systems",
    "Fortran": "First high-level language, still used in scientific computing and HPC",
    "Lisp": "Family of languages with symbolic expression syntax and homoiconicity",
    "Prolog": "Logic programming language based on formal logic and unification",
    "SQL": "Declarative language for querying and managing relational databases",
    "Bash": "Unix shell scripting language for command-line automation",
    "PowerShell": "Microsoft's task automation shell and scripting language",
    "Groovy": "Dynamic JVM language with syntax compatible with Java",
    "Visual Basic": "Event-driven language by Microsoft for rapid application development",
    "Delphi": "Object Pascal dialect for rapid application development",
    "Ada": "Strongly typed language designed for safety-critical real-time systems",
    "VHDL": "Hardware description language for FPGA and ASIC design",
    "Verilog": "Hardware description language for digital circuit design",
    "SystemVerilog": "Extension of Verilog adding verification and design features",
    "Scheme": "Minimalist Lisp dialect with lexical scoping and tail-call optimization",
    "Racket": "Language-oriented programming language descended from Scheme",
    "Smalltalk": "Pure object-oriented language where everything is an object and a message",
    "APL": "Array programming language using special mathematical notation",
    "Forth": "Stack-based extensible programming language",
    "Zig": "Systems language aiming to replace C with safety and modern features",
    "Nim": "Compiled systems language with Python-like syntax and metaprogramming",
    "Crystal": "Compiled language with Ruby-like syntax and static type checking",
    "V": "Simple compiled language inspired by Go, Rust, and Swift",
    "Hack": "Facebook's gradually typed language for the HHVM",
    "D": "Systems language combining C/C++ power with modern convenience",
    "Solidity": "Contract-oriented language for writing Ethereum smart contracts",
    "WASM": "WebAssembly — portable binary instruction format for stack-based virtual machine",
    "GraphQL": "Query language for APIs providing exactly requested data",
    "Tcl": "Dynamic scripting language for rapid prototyping and embedded use",
    "Awk": "Pattern scanning and text processing language for Unix",
    "Sed": "Stream editor for filtering and transforming text in Unix pipelines",
    "XSLT": "Transformation language for converting XML documents",
    "LaTeX": "Document preparation system and typesetting language for scientific publishing",
    "Markdown": "Lightweight markup language for formatting plain text",
    "YAML": "Human-readable data serialization format used in configuration",
    "JSON": "Lightweight data interchange format based on JavaScript object notation",
    "XML": "Extensible markup language for structured data representation",
    "HTML": "HyperText Markup Language — standard markup for web pages",
    "CSS": "Cascading Style Sheets — style sheet language for web presentation",
    "Sass": "CSS preprocessor adding variables, nesting, and mixins",
    "LLVM IR": "Low-level virtual machine intermediate representation for compiler backends",
    "WebGL": "JavaScript API for rendering 2D/3D graphics in the browser via OpenGL ES",
    "CUDA": "NVIDIA's parallel computing platform and programming model for GPUs",
    "OpenCL": "Open standard for cross-platform parallel programming on heterogeneous hardware",
    "MPI": "Message Passing Interface standard for parallel computing communication",
}

for name, defn in langs.items():
    C(name, defn)
    R(name, "Programming Language", "IS_A")

C("Programming Language", "Formal language comprising instructions for computation", "DEFINITION")
C("Compiled Language", "Language translated to machine code before execution")
C("Interpreted Language", "Language executed line by line at runtime")
C("Functional Programming", "Paradigm treating computation as evaluation of mathematical functions")
C("Object-Oriented Programming", "Paradigm organizing code into objects with state and behavior")
C("Imperative Programming", "Paradigm using statements that change program state")
C("Declarative Programming", "Paradigm describing what to compute, not how")
C("Logic Programming", "Paradigm expressing computation as logical relations")
C("Concurrent Programming", "Paradigm for executing multiple computations simultaneously")
C("Metaprogramming", "Writing programs that manipulate other programs or themselves")

R("C", "Compiled Language", "IS_A")
R("C++", "Compiled Language", "IS_A")
R("Rust", "Compiled Language", "IS_A")
R("Go", "Compiled Language", "IS_A")
R("Java", "Compiled Language", "IS_A")
R("Python", "Interpreted Language", "IS_A")
R("JavaScript", "Interpreted Language", "IS_A")
R("Ruby", "Interpreted Language", "IS_A")
R("Haskell", "Functional Programming", "USED_IN")
R("Erlang", "Functional Programming", "USED_IN")
R("Scala", "Functional Programming", "USED_IN")
R("F#", "Functional Programming", "USED_IN")
R("OCaml", "Functional Programming", "USED_IN")
R("Clojure", "Functional Programming", "USED_IN")
R("Java", "Object-Oriented Programming", "USED_IN")
R("C#", "Object-Oriented Programming", "USED_IN")
R("C++", "Object-Oriented Programming", "USED_IN")
R("Python", "Object-Oriented Programming", "USED_IN")
R("Prolog", "Logic Programming", "USED_IN")
R("SQL", "Declarative Programming", "USED_IN")
R("Go", "Concurrent Programming", "ENABLES")
R("Erlang", "Concurrent Programming", "ENABLES")
R("Rust", "Concurrent Programming", "ENABLES")
R("TypeScript", "JavaScript", "RELATES_TO")
R("Kotlin", "Java", "RELATES_TO")
R("Elixir", "Erlang", "RELATES_TO")
R("C++", "C", "RELATES_TO")
R("Objective-C", "C", "RELATES_TO")
R("Swift", "Objective-C", "RELATES_TO")
R("Dart", "Flutter", "USED_IN")
R("Solidity", "Ethereum", "USED_IN")
R("CUDA", "GPU", "USED_IN")

# ============================================================
# DATA STRUCTURES (60+)
# ============================================================
ds = {
    "Array": "Contiguous fixed-size collection of elements accessed by index in O(1)",
    "Dynamic Array": "Resizable array that doubles capacity when full, amortized O(1) append",
    "Linked List": "Linear collection where each node points to the next",
    "Singly Linked List": "Linked list where each node has a pointer to the next node only",
    "Doubly Linked List": "Linked list where each node has pointers to both next and previous",
    "Circular Linked List": "Linked list where the last node points back to the first",
    "Stack": "LIFO data structure supporting push and pop in O(1)",
    "Queue": "FIFO data structure supporting enqueue and dequeue in O(1)",
    "Deque": "Double-ended queue allowing insertion and removal at both ends",
    "Priority Queue": "Queue where elements are dequeued by priority, often via heap",
    "Heap": "Complete binary tree satisfying the heap property for efficient priority access",
    "Min-Heap": "Heap where parent is smaller than children, minimum at root",
    "Max-Heap": "Heap where parent is larger than children, maximum at root",
    "Binary Heap": "Heap implemented as complete binary tree in an array",
    "Fibonacci Heap": "Heap with amortized O(1) insert and decrease-key operations",
    "Binomial Heap": "Heap made of binomial trees supporting efficient merge",
    "Binary Tree": "Tree where each node has at most two children",
    "Binary Search Tree": "Binary tree where left < parent < right for ordered access",
    "AVL Tree": "Self-balancing BST maintaining height difference ≤ 1 between subtrees",
    "Red-Black Tree": "Self-balancing BST using node coloring to ensure O(log n) operations",
    "B-Tree": "Self-balancing tree optimized for disk access with high branching factor",
    "B+ Tree": "B-tree variant with all data in leaves and linked leaf nodes for range queries",
    "Splay Tree": "Self-adjusting BST that moves accessed elements to root",
    "Treap": "Randomized BST combining tree and heap properties",
    "Trie": "Tree for storing strings where each edge represents a character",
    "Radix Tree": "Compressed trie where single-child nodes are merged",
    "Suffix Tree": "Trie of all suffixes of a string for fast substring operations",
    "Suffix Array": "Sorted array of all suffixes of a string, space-efficient alternative to suffix tree",
    "Hash Table": "Key-value store using hash function for O(1) average lookup",
    "Hash Map": "Hash table implementation mapping keys to values",
    "Hash Set": "Hash table storing unique elements with O(1) membership test",
    "Bloom Filter": "Probabilistic data structure for set membership with false positives but no false negatives",
    "Cuckoo Filter": "Space-efficient probabilistic filter supporting deletion unlike Bloom filters",
    "Count-Min Sketch": "Probabilistic data structure for frequency estimation in streaming data",
    "HyperLogLog": "Probabilistic cardinality estimator using logarithmic space",
    "Graph": "Collection of vertices connected by edges representing relationships",
    "Directed Graph": "Graph where edges have direction from source to target",
    "Undirected Graph": "Graph where edges have no direction",
    "Weighted Graph": "Graph where edges carry numeric weights or costs",
    "Adjacency Matrix": "2D array representation of graph connectivity",
    "Adjacency List": "List-based representation of graph storing neighbors per vertex",
    "Disjoint Set": "Union-Find data structure for tracking partitioned sets with near O(1) operations",
    "Segment Tree": "Tree for efficient range queries and point updates in O(log n)",
    "Fenwick Tree": "Binary Indexed Tree for cumulative frequency queries in O(log n)",
    "Interval Tree": "Tree for storing intervals and finding overlapping intervals",
    "K-D Tree": "Space-partitioning tree for organizing points in k-dimensional space",
    "R-Tree": "Tree for indexing multi-dimensional spatial data like rectangles",
    "Quad Tree": "Tree partitioning 2D space into four quadrants recursively",
    "Octree": "Tree partitioning 3D space into eight octants recursively",
    "Skip List": "Probabilistic layered linked list providing O(log n) search",
    "Rope": "Binary tree of strings for efficient text editing operations",
    "Circular Buffer": "Fixed-size buffer that wraps around, used in streaming",
    "Sparse Table": "Preprocessed table for O(1) range minimum/maximum queries on static arrays",
    "Van Emde Boas Tree": "Tree supporting O(log log u) operations on integer universe",
    "Persistent Data Structure": "Data structure preserving previous versions after modification",
    "Immutable Data Structure": "Data structure that cannot be modified after creation",
    "LRU Cache": "Cache evicting least recently used items when full",
    "Ring Buffer": "Fixed circular buffer for producer-consumer scenarios",
    "Bitmap": "Array of bits for compact boolean storage and set operations",
    "Matrix": "2D array of numbers supporting linear algebra operations",
    "Sparse Matrix": "Matrix representation optimized for mostly-zero entries",
    "Tensor": "Multi-dimensional array generalizing scalars, vectors, and matrices",
}

C("Data Structure", "Way of organizing and storing data for efficient access and modification", "DEFINITION")
for name, defn in ds.items():
    C(name, defn)
    R(name, "Data Structure", "IS_A")

# DS relations
R("Dynamic Array", "Array", "IS_A")
R("Singly Linked List", "Linked List", "IS_A")
R("Doubly Linked List", "Linked List", "IS_A")
R("Circular Linked List", "Linked List", "IS_A")
R("Stack", "Array", "USED_IN")
R("Queue", "Linked List", "USED_IN")
R("Priority Queue", "Heap", "USED_IN")
R("Min-Heap", "Heap", "IS_A")
R("Max-Heap", "Heap", "IS_A")
R("Binary Heap", "Heap", "IS_A")
R("Fibonacci Heap", "Heap", "IS_A")
R("Binomial Heap", "Heap", "IS_A")
R("AVL Tree", "Binary Search Tree", "IS_A")
R("Red-Black Tree", "Binary Search Tree", "IS_A")
R("Splay Tree", "Binary Search Tree", "IS_A")
R("Treap", "Binary Search Tree", "IS_A")
R("B+ Tree", "B-Tree", "IS_A")
R("Binary Search Tree", "Binary Tree", "IS_A")
R("Trie", "Binary Tree", "RELATES_TO")
R("Radix Tree", "Trie", "IS_A")
R("Suffix Tree", "Trie", "IS_A")
R("Hash Map", "Hash Table", "IS_A")
R("Hash Set", "Hash Table", "IS_A")
R("Bloom Filter", "Hash Table", "RELATES_TO")
R("Segment Tree", "Binary Tree", "IS_A")
R("Fenwick Tree", "Binary Tree", "IS_A")
R("K-D Tree", "Binary Tree", "IS_A")
R("Quad Tree", "Binary Tree", "RELATES_TO")
R("LRU Cache", "Hash Map", "USED_IN")
R("LRU Cache", "Doubly Linked List", "USED_IN")
R("Adjacency Matrix", "Graph", "USED_IN")
R("Adjacency List", "Graph", "USED_IN")
R("Directed Graph", "Graph", "IS_A")
R("Undirected Graph", "Graph", "IS_A")
R("Weighted Graph", "Graph", "IS_A")
R("Sparse Matrix", "Matrix", "IS_A")
R("Tensor", "Matrix", "RELATES_TO")
R("Skip List", "Linked List", "RELATES_TO")

# ============================================================
# ALGORITHMS (120+)
# ============================================================
C("Algorithm", "Finite sequence of well-defined instructions to solve a class of problems", "DEFINITION")
C("Time Complexity", "Measure of computation time as function of input size", "DEFINITION")
C("Space Complexity", "Measure of memory usage as function of input size", "DEFINITION")
C("Big O Notation", "Asymptotic upper bound describing worst-case growth rate", "DEFINITION")

# Sorting
sorts = {
    "Bubble Sort": ("Simple comparison sort repeatedly swapping adjacent elements, O(n²)", 0.95),
    "Selection Sort": ("Sort finding minimum and placing at front each iteration, O(n²)", 0.95),
    "Insertion Sort": ("Sort inserting each element into its correct position, O(n²) but fast on small inputs", 0.95),
    "Merge Sort": ("Divide-and-conquer stable sort with O(n log n) guaranteed time", 0.97),
    "Quick Sort": ("Divide-and-conquer sort with O(n log n) average, O(n²) worst case", 0.97),
    "Heap Sort": ("Comparison sort using a heap for O(n log n) in-place sorting", 0.96),
    "Counting Sort": ("Non-comparison integer sort in O(n+k) time using counting array", 0.95),
    "Radix Sort": ("Non-comparison sort processing digits from least to most significant, O(nk)", 0.95),
    "Bucket Sort": ("Distribution sort placing elements into buckets then sorting each, O(n+k) average", 0.95),
    "Tim Sort": ("Hybrid merge/insertion sort used in Python and Java, O(n log n)", 0.96),
    "Shell Sort": ("Generalization of insertion sort using gap sequences", 0.94),
    "Intro Sort": ("Hybrid quick/heap/insertion sort used in C++ STL", 0.95),
    "Pigeonhole Sort": ("Sort for small range integers using pigeonhole principle", 0.93),
    "Cycle Sort": ("In-place sort minimizing memory writes, O(n²)", 0.93),
    "Patience Sort": ("Sort related to longest increasing subsequence, O(n log n)", 0.93),
    "Tree Sort": ("Sort by building a BST and doing in-order traversal", 0.93),
    "Bitonic Sort": ("Parallel-friendly comparison sort for hardware implementation", 0.93),
    "Pancake Sort": ("Sort using only prefix reversals as the operation", 0.92),
    "Bogo Sort": ("Randomly shuffles until sorted, expected O(n·n!) — intentionally terrible", 0.90),
    "Topological Sort": ("Linear ordering of directed acyclic graph vertices respecting edge direction", 0.96),
}
C("Sorting Algorithm", "Algorithm that arranges elements in a specific order")
for name, (defn, t) in sorts.items():
    C(name, defn, "DEFINITION", t)
    R(name, "Sorting Algorithm", "IS_A")
    R(name, "Algorithm", "IS_A")

R("Merge Sort", "Divide and Conquer", "USED_IN")
R("Quick Sort", "Divide and Conquer", "USED_IN")
R("Heap Sort", "Heap", "USED_IN")
R("Tim Sort", "Merge Sort", "RELATES_TO")
R("Tim Sort", "Insertion Sort", "RELATES_TO")
R("Intro Sort", "Quick Sort", "RELATES_TO")
R("Intro Sort", "Heap Sort", "RELATES_TO")
R("Counting Sort", "Radix Sort", "USED_IN")
R("Topological Sort", "Directed Graph", "USED_IN")

# Searching
searches = {
    "Linear Search": "Sequential scan through elements, O(n)",
    "Binary Search": "Divide-and-conquer search on sorted array, O(log n)",
    "Interpolation Search": "Search estimating position based on value distribution, O(log log n) average",
    "Exponential Search": "Search combining exponential jumps with binary search",
    "Jump Search": "Search jumping √n steps then linear scanning, O(√n)",
    "Ternary Search": "Search dividing range into thirds for unimodal functions",
    "Fibonacci Search": "Search using Fibonacci numbers to divide the search interval",
    "Depth-First Search": "Graph traversal exploring as deep as possible before backtracking",
    "Breadth-First Search": "Graph traversal exploring all neighbors before going deeper",
    "Iterative Deepening DFS": "DFS with increasing depth limits combining BFS completeness with DFS space",
    "A* Search": "Best-first graph search using heuristic to find shortest path optimally",
    "Uniform Cost Search": "Graph search expanding cheapest node first, optimal for non-negative costs",
    "Bidirectional Search": "Search from both start and goal meeting in the middle",
    "Best-First Search": "Graph search selecting most promising node by heuristic evaluation",
}
C("Search Algorithm", "Algorithm for finding elements or paths in a data structure")
for name, defn in searches.items():
    C(name, defn)
    R(name, "Search Algorithm", "IS_A")
    R(name, "Algorithm", "IS_A")

R("Binary Search", "Array", "USED_IN")
R("Depth-First Search", "Graph", "USED_IN")
R("Breadth-First Search", "Graph", "USED_IN")
R("Depth-First Search", "Stack", "USED_IN")
R("Breadth-First Search", "Queue", "USED_IN")
R("A* Search", "Priority Queue", "USED_IN")

# Graph algorithms
graphs = {
    "Dijkstra's Algorithm": "Shortest path from single source with non-negative weights, O((V+E) log V)",
    "Bellman-Ford Algorithm": "Single-source shortest path handling negative weights, O(VE)",
    "Floyd-Warshall Algorithm": "All-pairs shortest paths via dynamic programming, O(V³)",
    "Kruskal's Algorithm": "Minimum spanning tree by sorting edges and using union-find, O(E log E)",
    "Prim's Algorithm": "Minimum spanning tree growing from a vertex using priority queue",
    "Borůvka's Algorithm": "Minimum spanning tree by parallel edge selection phases",
    "Tarjan's Algorithm": "Finding strongly connected components in O(V+E)",
    "Kosaraju's Algorithm": "Finding strongly connected components using two DFS passes",
    "Ford-Fulkerson Algorithm": "Maximum flow by finding augmenting paths",
    "Edmonds-Karp Algorithm": "Max flow using BFS for augmenting paths, O(VE²)",
    "Dinic's Algorithm": "Max flow using layered networks and blocking flows, O(V²E)",
    "Push-Relabel Algorithm": "Max flow maintaining preflow with push and relabel operations",
    "Hungarian Algorithm": "Optimal assignment in bipartite graphs in O(n³)",
    "Hopcroft-Karp Algorithm": "Maximum bipartite matching in O(E√V)",
    "Johnson's Algorithm": "All-pairs shortest paths using Bellman-Ford reweighting then Dijkstra",
    "Warshall's Algorithm": "Transitive closure of a directed graph in O(V³)",
    "Fleury's Algorithm": "Finding Eulerian paths by avoiding bridges",
    "Hierholzer's Algorithm": "Finding Eulerian circuits in O(E) time",
    "Articulation Point Algorithm": "Finding cut vertices whose removal disconnects a graph",
    "Bridge Finding Algorithm": "Finding edges whose removal disconnects a graph",
    "Graph Coloring": "Assigning colors to vertices so no adjacent vertices share a color",
    "Maximum Independent Set": "Finding largest set of vertices with no edges between them",
    "Minimum Vertex Cover": "Finding smallest set of vertices covering all edges",
    "Traveling Salesman Problem": "NP-hard problem of finding shortest tour visiting all cities",
    "Christofides Algorithm": "1.5-approximation for metric TSP using MST and matching",
    "PageRank": "Algorithm ranking graph nodes by iterative link-weight propagation",
    "Louvain Algorithm": "Community detection by modularity optimization in networks",
}
C("Graph Algorithm", "Algorithm operating on graph structures to solve connectivity and optimization problems")
for name, defn in graphs.items():
    C(name, defn)
    R(name, "Graph Algorithm", "IS_A")
    R(name, "Algorithm", "IS_A")

R("Dijkstra's Algorithm", "Priority Queue", "USED_IN")
R("Dijkstra's Algorithm", "Weighted Graph", "USED_IN")
R("Kruskal's Algorithm", "Disjoint Set", "USED_IN")
R("Prim's Algorithm", "Priority Queue", "USED_IN")
R("Bellman-Ford Algorithm", "Weighted Graph", "USED_IN")
R("Floyd-Warshall Algorithm", "Dynamic Programming", "USED_IN")
R("Ford-Fulkerson Algorithm", "Depth-First Search", "USED_IN")
R("Edmonds-Karp Algorithm", "Breadth-First Search", "USED_IN")
R("PageRank", "Google", "USED_IN")

# String algorithms
strings = {
    "KMP Algorithm": "Pattern matching with O(n+m) using failure function preprocessing",
    "Rabin-Karp Algorithm": "Pattern matching using rolling hash for average O(n+m)",
    "Boyer-Moore Algorithm": "Efficient string search using bad character and good suffix rules",
    "Aho-Corasick Algorithm": "Multi-pattern string matching using automaton, O(n+m+z)",
    "Z Algorithm": "Linear time pattern matching using Z-array",
    "Manacher's Algorithm": "Finding all palindromic substrings in O(n)",
    "Levenshtein Distance": "Minimum edit distance between strings via dynamic programming",
    "Longest Common Subsequence": "Finding longest subsequence common to two sequences, O(nm)",
    "Longest Common Substring": "Finding longest contiguous substring common to two strings",
    "Longest Increasing Subsequence": "Finding longest monotonically increasing subsequence",
    "Knuth-Morris-Pratt": "Same as KMP, prefix-function based pattern matching",
}
C("String Algorithm", "Algorithm for processing and searching text strings")
for name, defn in strings.items():
    C(name, defn)
    R(name, "String Algorithm", "IS_A")
    R(name, "Algorithm", "IS_A")

R("KMP Algorithm", "Trie", "RELATES_TO")
R("Aho-Corasick Algorithm", "Trie", "USED_IN")
R("Levenshtein Distance", "Dynamic Programming", "USED_IN")
R("Longest Common Subsequence", "Dynamic Programming", "USED_IN")

# DP and other techniques
techniques = {
    "Dynamic Programming": "Optimization technique solving overlapping subproblems by memoization",
    "Greedy Algorithm": "Algorithm making locally optimal choices at each step",
    "Divide and Conquer": "Strategy splitting problems into subproblems, solving, and combining",
    "Backtracking": "Systematic search trying and abandoning partial solutions",
    "Branch and Bound": "Optimization using tree exploration with pruning by bounds",
    "Memoization": "Caching function results to avoid redundant computation",
    "Tabulation": "Bottom-up dynamic programming filling a table iteratively",
    "Recursion": "Technique where a function calls itself to solve smaller subproblems",
    "Tail Recursion": "Recursion where recursive call is the last operation, enabling optimization",
    "Amortized Analysis": "Averaging time over sequence of operations for tighter bounds",
    "Randomized Algorithm": "Algorithm using random choices to achieve good expected performance",
    "Approximation Algorithm": "Algorithm finding near-optimal solutions with provable guarantee",
    "Monte Carlo Algorithm": "Randomized algorithm that may give incorrect results with low probability",
    "Las Vegas Algorithm": "Randomized algorithm that always gives correct results with random runtime",
    "Online Algorithm": "Algorithm processing input sequentially without future knowledge",
    "Streaming Algorithm": "Algorithm processing data in one pass with limited memory",
    "Parallel Algorithm": "Algorithm designed to run on multiple processors simultaneously",
    "Distributed Algorithm": "Algorithm running across multiple networked machines",
    "Genetic Algorithm": "Optimization using evolution-inspired selection, crossover, and mutation",
    "Simulated Annealing": "Probabilistic optimization gradually reducing randomness to find global optimum",
    "Gradient Descent": "Iterative optimization following negative gradient to minimize a function",
    "Stochastic Gradient Descent": "Gradient descent using random subset of data per step",
    "Newton's Method": "Root-finding and optimization using second-order Taylor approximation",
    "Bisection Method": "Root-finding by repeatedly halving an interval",
    "Simplex Algorithm": "Linear programming optimization traversing polytope vertices",
    "Interior Point Method": "Linear programming traversing the interior of the feasible region",
    "Branch and Cut": "Combination of branch and bound with cutting planes for integer programming",
    "Minimax Algorithm": "Game tree search minimizing opponent's maximum gain",
    "Alpha-Beta Pruning": "Minimax optimization pruning branches that cannot affect the decision",
    "MCTS": "Monte Carlo Tree Search — sampling-based game tree search used in AlphaGo",
}
for name, defn in techniques.items():
    C(name, defn)
    R(name, "Algorithm", "IS_A" if "Algorithm" in name else "RELATES_TO")

R("Memoization", "Dynamic Programming", "USED_IN")
R("Tabulation", "Dynamic Programming", "USED_IN")
R("Alpha-Beta Pruning", "Minimax Algorithm", "RELATES_TO")
R("Stochastic Gradient Descent", "Gradient Descent", "IS_A")
R("Genetic Algorithm", "Randomized Algorithm", "IS_A")
R("Monte Carlo Algorithm", "Randomized Algorithm", "IS_A")
R("Las Vegas Algorithm", "Randomized Algorithm", "IS_A")
R("MCTS", "Monte Carlo Algorithm", "RELATES_TO")

# Numeric / crypto
numerics = {
    "Fast Fourier Transform": "Computing DFT in O(n log n) using divide and conquer",
    "Euclidean Algorithm": "Finding GCD of two numbers by repeated division",
    "Extended Euclidean Algorithm": "Finding GCD plus coefficients satisfying Bézout's identity",
    "Sieve of Eratosthenes": "Finding all primes up to n by iteratively marking composites",
    "Miller-Rabin Primality Test": "Probabilistic primality test based on Fermat's little theorem",
    "RSA Algorithm": "Public-key cryptography based on difficulty of factoring large primes",
    "AES Algorithm": "Symmetric block cipher standard using substitution-permutation network",
    "SHA-256": "Cryptographic hash function producing 256-bit digest",
    "Diffie-Hellman Key Exchange": "Protocol for secure key agreement over insecure channel",
    "Elliptic Curve Cryptography": "Public-key cryptography using elliptic curve discrete logarithm",
    "Gaussian Elimination": "Algorithm for solving systems of linear equations by row reduction",
    "LU Decomposition": "Factoring matrix into lower and upper triangular matrices",
    "QR Decomposition": "Factoring matrix into orthogonal and upper triangular matrices",
    "Singular Value Decomposition": "Factoring matrix into U·Σ·Vᵀ for dimensionality reduction",
    "Principal Component Analysis": "Dimensionality reduction finding directions of maximum variance",
    "K-Means Clustering": "Partitioning n points into k clusters by minimizing within-cluster variance",
    "DBSCAN": "Density-based clustering finding arbitrarily shaped clusters",
    "Random Forest": "Ensemble of decision trees using bagging for classification and regression",
    "Gradient Boosting": "Ensemble method building models sequentially to correct previous errors",
    "Support Vector Machine": "Classifier finding optimal hyperplane maximizing margin",
    "Neural Network": "Computing model of interconnected nodes inspired by biological neurons",
    "Backpropagation": "Algorithm computing gradients for training neural networks via chain rule",
    "Convolutional Neural Network": "Neural network using convolution layers for spatial data like images",
    "Recurrent Neural Network": "Neural network with loops for sequential data processing",
    "Transformer": "Neural architecture using self-attention for parallel sequence processing",
    "Attention Mechanism": "Neural module weighting input relevance dynamically",
    "GAN": "Generative Adversarial Network — two networks competing to generate realistic data",
    "Autoencoder": "Neural network learning compressed representations by encoding and decoding",
    "Variational Autoencoder": "Generative autoencoder with probabilistic latent space",
    "Diffusion Model": "Generative model learning to denoise data from Gaussian noise iteratively",
    "Decision Tree": "Tree-structured classifier splitting on features for predictions",
    "Naive Bayes": "Probabilistic classifier assuming feature independence given class",
    "Logistic Regression": "Linear model for binary classification using sigmoid function",
    "Linear Regression": "Model fitting linear relationship between variables by least squares",
    "K-Nearest Neighbors": "Classification by majority vote of k closest training examples",
}
for name, defn in numerics.items():
    C(name, defn)
    R(name, "Algorithm", "IS_A")

R("Backpropagation", "Neural Network", "USED_IN")
R("Backpropagation", "Gradient Descent", "USED_IN")
R("Convolutional Neural Network", "Neural Network", "IS_A")
R("Recurrent Neural Network", "Neural Network", "IS_A")
R("Transformer", "Attention Mechanism", "USED_IN")
R("GAN", "Neural Network", "IS_A")
R("Autoencoder", "Neural Network", "IS_A")
R("Variational Autoencoder", "Autoencoder", "IS_A")
R("Diffusion Model", "Neural Network", "IS_A")
R("Random Forest", "Decision Tree", "USED_IN")
R("Gradient Boosting", "Decision Tree", "USED_IN")
R("Principal Component Analysis", "Singular Value Decomposition", "USED_IN")
R("RSA Algorithm", "Euclidean Algorithm", "USED_IN")
R("SHA-256", "Hash Table", "RELATES_TO")

# ============================================================
# SOFTWARE TOOLS & FRAMEWORKS (200+)
# ============================================================
tools = {
    # Web frameworks
    "React": "JavaScript library for building user interfaces with component-based architecture",
    "Angular": "Google's TypeScript-based web application framework",
    "Vue.js": "Progressive JavaScript framework for building UIs",
    "Svelte": "Compile-time web framework with no virtual DOM",
    "Next.js": "React framework with server-side rendering and static generation",
    "Nuxt.js": "Vue.js framework for server-side rendered applications",
    "Remix": "Full-stack React framework focused on web standards",
    "Astro": "Static site framework delivering zero JavaScript by default",
    "Express.js": "Minimal Node.js web application framework",
    "Fastify": "High-performance Node.js web framework",
    "Koa": "Next-generation Node.js web framework by Express creators",
    "NestJS": "Progressive Node.js framework for scalable server-side applications",
    "Django": "Python high-level web framework encouraging rapid development",
    "Flask": "Python lightweight micro web framework",
    "FastAPI": "Modern Python web framework for building APIs with type hints",
    "Ruby on Rails": "Ruby web framework emphasizing convention over configuration",
    "Sinatra": "Lightweight Ruby web framework for simple applications",
    "Spring Boot": "Java framework for creating production-ready Spring applications",
    "Spring Framework": "Comprehensive Java enterprise application framework",
    "Micronaut": "JVM framework for building modular microservices",
    "Quarkus": "Kubernetes-native Java framework with fast startup",
    "Laravel": "PHP web framework with elegant syntax and MVC architecture",
    "Symfony": "PHP web framework and set of reusable components",
    "ASP.NET Core": "Cross-platform .NET framework for building web apps and APIs",
    "Blazor": ".NET framework for building interactive web UIs with C#",
    "Gin": "High-performance HTTP web framework for Go",
    "Echo": "High-performance Go web framework",
    "Fiber": "Express-inspired Go web framework built on fasthttp",
    "Actix Web": "Powerful Rust web framework with actor-based architecture",
    "Rocket": "Rust web framework focusing on ease of use and type safety",
    "Axum": "Ergonomic Rust web framework built on Tokio and Tower",
    "Phoenix": "Elixir web framework for productive, reliable applications",
    "Flutter": "Google's UI toolkit for building natively compiled cross-platform apps",
    "React Native": "Framework for building native mobile apps using React",
    "Electron": "Framework for building cross-platform desktop apps with web technologies",
    "Tauri": "Lightweight alternative to Electron using system webview and Rust backend",
    "Qt": "Cross-platform C++ framework for desktop and embedded applications",
    "GTK": "Cross-platform widget toolkit for creating graphical user interfaces",
    "SwiftUI": "Apple's declarative UI framework for all Apple platforms",
    "Jetpack Compose": "Android's modern declarative UI toolkit",
    # Data & ML
    "TensorFlow": "Google's open-source machine learning framework",
    "PyTorch": "Facebook's open-source machine learning framework with dynamic computation graphs",
    "JAX": "Google's library for high-performance numerical computing and ML research",
    "Keras": "High-level neural network API running on TensorFlow",
    "scikit-learn": "Python library for classical machine learning algorithms",
    "Pandas": "Python data manipulation and analysis library with DataFrame",
    "NumPy": "Python library for numerical computing with multi-dimensional arrays",
    "SciPy": "Python library for scientific and technical computing",
    "Matplotlib": "Python 2D plotting library for data visualization",
    "Seaborn": "Statistical data visualization library built on Matplotlib",
    "Plotly": "Interactive graphing library for Python, R, and JavaScript",
    "D3.js": "JavaScript library for producing dynamic data visualizations in the browser",
    "Apache Spark": "Distributed computing framework for big data processing",
    "Apache Hadoop": "Framework for distributed storage and processing of big data",
    "Apache Kafka": "Distributed event streaming platform for high-throughput data pipelines",
    "Apache Flink": "Stream processing framework for stateful computations over data streams",
    "Apache Airflow": "Platform for programmatically authoring and scheduling data workflows",
    "dbt": "Data transformation tool using SQL for analytics engineering",
    "Snowflake": "Cloud-native data warehouse with separation of storage and compute",
    "Databricks": "Unified analytics platform built on Apache Spark",
    "Hugging Face": "Platform and library for state-of-the-art NLP models",
    "LangChain": "Framework for developing applications powered by language models",
    "MLflow": "Platform for managing the ML lifecycle including experiments and deployment",
    "Kubeflow": "ML toolkit for Kubernetes for deploying ML workflows",
    "ONNX": "Open format for representing machine learning models across frameworks",
    "OpenCV": "Computer vision library with optimized algorithms for image processing",
    "spaCy": "Industrial-strength NLP library for Python",
    "NLTK": "Python platform for natural language processing research and education",
    # Databases
    "PostgreSQL": "Advanced open-source relational database with extensibility",
    "MySQL": "Popular open-source relational database management system",
    "MariaDB": "Community fork of MySQL with additional features",
    "SQLite": "Self-contained serverless SQL database engine",
    "Oracle Database": "Enterprise relational database management system by Oracle",
    "SQL Server": "Microsoft's enterprise relational database management system",
    "MongoDB": "Document-oriented NoSQL database storing BSON documents",
    "Redis": "In-memory key-value store used as database, cache, and message broker",
    "Cassandra": "Distributed wide-column NoSQL database for high availability",
    "DynamoDB": "AWS fully managed NoSQL key-value and document database",
    "Elasticsearch": "Distributed search and analytics engine based on Apache Lucene",
    "Neo4j": "Graph database using Cypher query language for connected data",
    "CouchDB": "Document-oriented NoSQL database with HTTP API and replication",
    "InfluxDB": "Time-series database optimized for fast writes and queries",
    "ClickHouse": "Column-oriented OLAP database for real-time analytics",
    "Supabase": "Open-source Firebase alternative built on PostgreSQL",
    "PlanetScale": "Serverless MySQL platform with Vitess-based horizontal scaling",
    "CockroachDB": "Distributed SQL database surviving disk, machine, and datacenter failures",
    "TimescaleDB": "Time-series database extension for PostgreSQL",
    "Memcached": "Distributed memory caching system for speeding up web applications",
    "Druid": "Real-time analytics database for fast aggregation queries on event data",
    # DevOps & Infrastructure
    "Docker": "Platform for building and running applications in containers",
    "Kubernetes": "Container orchestration platform for automating deployment and scaling",
    "Terraform": "Infrastructure as code tool for provisioning cloud resources",
    "Ansible": "Agentless IT automation tool for configuration management",
    "Puppet": "Configuration management tool for infrastructure automation",
    "Chef": "Configuration management tool using Ruby DSL",
    "Vagrant": "Tool for building and managing virtual machine environments",
    "Helm": "Package manager for Kubernetes applications",
    "Istio": "Service mesh providing traffic management and security for microservices",
    "Envoy": "High-performance proxy for service mesh and edge networking",
    "Nginx": "High-performance web server, reverse proxy, and load balancer",
    "Apache HTTP Server": "Widely used open-source web server",
    "Caddy": "Web server with automatic HTTPS via Let's Encrypt",
    "HAProxy": "High-availability load balancer and TCP/HTTP proxy",
    "Traefik": "Modern reverse proxy and load balancer for microservices",
    "Prometheus": "Monitoring system with time-series database and alerting",
    "Grafana": "Open-source platform for monitoring and observability dashboards",
    "Datadog": "Cloud monitoring and security platform for infrastructure and applications",
    "New Relic": "Observability platform for application performance monitoring",
    "Splunk": "Platform for searching, monitoring, and analyzing machine-generated data",
    "ELK Stack": "Elasticsearch, Logstash, Kibana — log management and analytics stack",
    "Jaeger": "Distributed tracing system for monitoring microservice architectures",
    "OpenTelemetry": "Observability framework for traces, metrics, and logs",
    "Jenkins": "Open-source automation server for CI/CD pipelines",
    "GitHub Actions": "CI/CD automation integrated into GitHub repositories",
    "GitLab CI": "Continuous integration and delivery built into GitLab",
    "CircleCI": "Cloud-based CI/CD platform for automating build and deploy",
    "ArgoCD": "Declarative GitOps continuous delivery tool for Kubernetes",
    "Flux": "GitOps toolkit for keeping Kubernetes clusters in sync with sources",
    "Pulumi": "Infrastructure as code using general-purpose programming languages",
    "AWS CloudFormation": "AWS infrastructure as code service using templates",
    "Packer": "Tool for building identical machine images for multiple platforms",
    "Vault": "HashiCorp's secrets management and data protection tool",
    "Consul": "Service mesh and service discovery tool by HashiCorp",
    # Version control
    "Git": "Distributed version control system for tracking source code changes",
    "GitHub": "Web-based Git repository hosting with collaboration features",
    "GitLab": "DevOps platform with Git repository management and CI/CD",
    "Bitbucket": "Atlassian's Git repository hosting with Jira integration",
    "SVN": "Apache Subversion — centralized version control system",
    "Mercurial": "Distributed version control system designed for large projects",
    # IDEs & Editors
    "VS Code": "Microsoft's lightweight extensible source code editor",
    "IntelliJ IDEA": "JetBrains IDE for Java and JVM language development",
    "PyCharm": "JetBrains IDE specialized for Python development",
    "WebStorm": "JetBrains IDE for JavaScript and TypeScript development",
    "Eclipse": "Open-source IDE primarily for Java development",
    "Xcode": "Apple's IDE for macOS and iOS application development",
    "Android Studio": "Google's official IDE for Android development",
    "Vim": "Highly configurable modal text editor for efficient text editing",
    "Neovim": "Modernized fork of Vim with Lua extensibility and async support",
    "Emacs": "Extensible, customizable text editor with Lisp-based configuration",
    "Sublime Text": "Fast, lightweight text editor with powerful features",
    "Atom": "Hackable text editor by GitHub (now sunset)",
    "JupyterLab": "Web-based interactive development environment for notebooks",
    "RStudio": "IDE for R programming and statistical computing",
    # Testing
    "Jest": "JavaScript testing framework with snapshot testing",
    "Mocha": "JavaScript test framework for Node.js and browser",
    "Cypress": "End-to-end testing framework for web applications",
    "Playwright": "Cross-browser end-to-end testing framework by Microsoft",
    "Selenium": "Browser automation framework for web application testing",
    "JUnit": "Unit testing framework for Java applications",
    "pytest": "Python testing framework with simple syntax and plugins",
    "RSpec": "Behavior-driven development testing framework for Ruby",
    "Postman": "API development and testing platform",
    "k6": "Modern load testing tool for performance testing",
    "Locust": "Python-based load testing framework",
    # Cloud
    "AWS": "Amazon Web Services — comprehensive cloud computing platform",
    "Azure": "Microsoft's cloud computing platform and services",
    "Google Cloud Platform": "Google's suite of cloud computing services",
    "Vercel": "Platform for frontend deployment with serverless functions",
    "Netlify": "Platform for deploying and hosting static sites and serverless functions",
    "Cloudflare": "CDN, security, and edge computing platform",
    "DigitalOcean": "Cloud infrastructure provider for developers",
    "Heroku": "Platform as a service for deploying and managing applications",
    "Fly.io": "Platform for deploying full-stack apps globally at the edge",
    "Railway": "Platform for deploying infrastructure with zero config",
    # Package managers
    "npm": "Node.js package manager and registry",
    "Yarn": "Fast, reliable JavaScript package manager",
    "pnpm": "Fast, disk-efficient JavaScript package manager",
    "pip": "Python package installer from PyPI",
    "Cargo": "Rust package manager and build system",
    "Maven": "Java project management and build automation tool",
    "Gradle": "Build automation tool for JVM projects using Groovy/Kotlin DSL",
    "NuGet": ".NET package manager",
    "Homebrew": "Package manager for macOS and Linux",
    "apt": "Debian/Ubuntu package management tool",
    "Composer": "Dependency manager for PHP",
    # Message queues & comm
    "RabbitMQ": "Open-source message broker implementing AMQP protocol",
    "ZeroMQ": "High-performance asynchronous messaging library",
    "NATS": "Lightweight cloud-native messaging system",
    "gRPC": "High-performance RPC framework using Protocol Buffers",
    "Protocol Buffers": "Google's language-neutral data serialization format",
    "Apache Thrift": "Framework for scalable cross-language services development",
    "GraphQL (Apollo)": "GraphQL implementation with client and server libraries",
    "REST": "Representational State Transfer — architectural style for web APIs",
    "WebSocket": "Protocol providing full-duplex communication over single TCP connection",
    "Socket.IO": "Library enabling real-time bidirectional event-based communication",
    # Security
    "OWASP": "Open Web Application Security Project — community for web security",
    "Burp Suite": "Web security testing platform for finding vulnerabilities",
    "Wireshark": "Network protocol analyzer for deep packet inspection",
    "Nmap": "Network discovery and security auditing tool",
    "Metasploit": "Penetration testing framework for finding security vulnerabilities",
    "Snort": "Open-source network intrusion detection and prevention system",
    "OpenSSL": "Toolkit implementing SSL/TLS and general-purpose cryptography",
    "Let's Encrypt": "Free automated certificate authority for TLS certificates",
    "Fail2Ban": "Intrusion prevention software monitoring log files for malicious activity",
    "ClamAV": "Open-source antivirus engine for detecting malicious software",
}

for name, defn in tools.items():
    C(name, defn)

# Tool categorization relations
web_fws = ["React", "Angular", "Vue.js", "Svelte", "Next.js", "Nuxt.js", "Django", "Flask", "FastAPI",
           "Ruby on Rails", "Spring Boot", "Laravel", "ASP.NET Core", "Gin", "Phoenix", "Actix Web", "Rocket", "Axum",
           "Express.js", "NestJS", "Fiber", "Echo", "Remix", "Astro", "Koa", "Fastify", "Sinatra",
           "Symfony", "Blazor", "Micronaut", "Quarkus"]
C("Web Framework", "Software framework designed for building web applications and APIs")
for fw in web_fws:
    R(fw, "Web Framework", "IS_A")

R("React", "JavaScript", "USED_IN"); R("Angular", "TypeScript", "USED_IN")
R("Vue.js", "JavaScript", "USED_IN"); R("Django", "Python", "USED_IN")
R("Flask", "Python", "USED_IN"); R("FastAPI", "Python", "USED_IN")
R("Ruby on Rails", "Ruby", "USED_IN"); R("Spring Boot", "Java", "USED_IN")
R("Laravel", "PHP", "USED_IN"); R("ASP.NET Core", "C#", "USED_IN")
R("Gin", "Go", "USED_IN"); R("Phoenix", "Elixir", "USED_IN")
R("Actix Web", "Rust", "USED_IN"); R("Rocket", "Rust", "USED_IN")
R("Next.js", "React", "USED_IN"); R("Nuxt.js", "Vue.js", "USED_IN")
R("Flutter", "Dart", "USED_IN"); R("React Native", "React", "USED_IN")

dbs = ["PostgreSQL", "MySQL", "MariaDB", "SQLite", "Oracle Database", "SQL Server",
       "MongoDB", "Redis", "Cassandra", "DynamoDB", "Elasticsearch", "Neo4j",
       "CouchDB", "InfluxDB", "ClickHouse", "CockroachDB", "TimescaleDB", "Memcached", "Druid"]
C("Database", "Organized collection of data stored and accessed electronically")
C("Relational Database", "Database organizing data into tables with relationships via SQL")
C("NoSQL Database", "Database using non-tabular models for flexible data storage")
for db in dbs:
    R(db, "Database", "IS_A")
for rdb in ["PostgreSQL", "MySQL", "MariaDB", "SQLite", "Oracle Database", "SQL Server", "CockroachDB"]:
    R(rdb, "Relational Database", "IS_A")
    R(rdb, "SQL", "USED_IN")
for ndb in ["MongoDB", "Redis", "Cassandra", "DynamoDB", "CouchDB", "Neo4j"]:
    R(ndb, "NoSQL Database", "IS_A")

ml_tools = ["TensorFlow", "PyTorch", "JAX", "Keras", "scikit-learn", "Hugging Face", "LangChain", "MLflow", "Kubeflow", "ONNX", "OpenCV", "spaCy", "NLTK"]
C("Machine Learning Framework", "Software library providing tools for building and training ML models")
for t in ml_tools:
    R(t, "Machine Learning Framework", "IS_A")
    R(t, "Python", "USED_IN")

R("TensorFlow", "Neural Network", "ENABLES")
R("PyTorch", "Neural Network", "ENABLES")
R("Keras", "TensorFlow", "USED_IN")

devops = ["Docker", "Kubernetes", "Terraform", "Ansible", "Puppet", "Chef", "Helm", "Istio", "Vagrant", "Pulumi", "Packer", "Vault", "Consul"]
C("DevOps Tool", "Tool supporting development operations, automation, and infrastructure management")
for t in devops:
    R(t, "DevOps Tool", "IS_A")

R("Kubernetes", "Docker", "USED_IN")
R("Helm", "Kubernetes", "USED_IN")
R("Istio", "Kubernetes", "USED_IN")
R("ArgoCD", "Kubernetes", "USED_IN")

ci_tools = ["Jenkins", "GitHub Actions", "GitLab CI", "CircleCI", "ArgoCD", "Flux"]
C("CI/CD Tool", "Tool automating continuous integration and deployment pipelines")
for t in ci_tools:
    R(t, "CI/CD Tool", "IS_A")

R("Prometheus", "Grafana", "USED_IN")
R("ELK Stack", "Elasticsearch", "USED_IN")
R("Docker", "Linux", "USED_IN")
R("Git", "GitHub", "USED_IN")
R("Git", "GitLab", "USED_IN")

cloud_providers = ["AWS", "Azure", "Google Cloud Platform", "DigitalOcean", "Cloudflare"]
C("Cloud Platform", "Platform providing on-demand computing resources and services over the internet")
for cp in cloud_providers:
    R(cp, "Cloud Platform", "IS_A")

# ============================================================
# COMPUTER HARDWARE (200+)
# ============================================================
hw = {
    # CPU
    "CPU": "Central Processing Unit — executes instructions and performs calculations",
    "ALU": "Arithmetic Logic Unit — performs arithmetic and bitwise operations",
    "Control Unit": "CPU component directing operation of the processor",
    "Register": "Small fast storage location inside the CPU",
    "Program Counter": "Register holding address of next instruction to execute",
    "Instruction Register": "Register holding the currently executing instruction",
    "Stack Pointer": "Register pointing to the top of the call stack",
    "Cache": "Small fast memory between CPU and main memory",
    "L1 Cache": "Fastest and smallest CPU cache, split into instruction and data",
    "L2 Cache": "Larger but slower CPU cache, per-core or shared",
    "L3 Cache": "Largest on-die cache shared across all CPU cores",
    "Pipeline": "CPU technique overlapping instruction execution stages",
    "Branch Predictor": "CPU unit predicting conditional branch outcomes to avoid stalls",
    "Out-of-Order Execution": "CPU executing instructions as operands become available, not in order",
    "Superscalar Architecture": "CPU architecture issuing multiple instructions per clock cycle",
    "SIMD": "Single Instruction Multiple Data — parallel processing of data arrays",
    "Hyper-Threading": "Intel's simultaneous multithreading running two threads per core",
    "Clock Speed": "Frequency at which CPU executes instructions, measured in GHz",
    "Instruction Set Architecture": "Abstract model of a CPU defining instructions and registers",
    "x86 Architecture": "CISC instruction set architecture by Intel/AMD for desktop and server CPUs",
    "ARM Architecture": "RISC instruction set architecture for power-efficient processors",
    "RISC-V": "Open-source RISC instruction set architecture",
    "MIPS Architecture": "RISC instruction set architecture used in embedded and academic settings",
    "CISC": "Complex Instruction Set Computer with many specialized instructions",
    "RISC": "Reduced Instruction Set Computer with simple, fast instructions",
    "Microcontroller": "Small computer on single chip with CPU, memory, and I/O peripherals",
    "FPGA": "Field-Programmable Gate Array — reconfigurable integrated circuit",
    "ASIC": "Application-Specific Integrated Circuit designed for a particular use",
    "SoC": "System on Chip — integrating CPU, GPU, memory controller on one die",
    "Chiplet": "Small modular die composing a larger processor package",
    "Die": "Single continuous piece of semiconductor with integrated circuits",
    "Wafer": "Thin slice of semiconductor material for fabricating integrated circuits",
    "Transistor": "Semiconductor device switching or amplifying electrical signals",
    "MOSFET": "Metal-Oxide-Semiconductor Field-Effect Transistor — fundamental building block",
    "Logic Gate": "Physical implementation of a Boolean function on one or more inputs",
    "AND Gate": "Logic gate outputting 1 only when all inputs are 1",
    "OR Gate": "Logic gate outputting 1 when any input is 1",
    "NOT Gate": "Logic gate inverting its input",
    "NAND Gate": "Universal logic gate — NOT AND, can build any other gate",
    "NOR Gate": "Universal logic gate — NOT OR",
    "XOR Gate": "Logic gate outputting 1 when inputs differ",
    "Flip-Flop": "Sequential circuit storing one bit of state",
    "Latch": "Level-triggered storage element holding a bit",
    "Multiplexer": "Circuit selecting one of many inputs to a single output",
    "Demultiplexer": "Circuit routing a single input to one of many outputs",
    "Adder": "Circuit performing binary addition",
    "Full Adder": "Adder computing sum and carry for three input bits",
    "Half Adder": "Adder computing sum and carry for two input bits",
    # Memory
    "RAM": "Random Access Memory — volatile memory for active data",
    "DRAM": "Dynamic RAM using capacitors, requires periodic refreshing",
    "SRAM": "Static RAM using flip-flops, faster but more expensive than DRAM",
    "DDR4": "Fourth generation double data rate synchronous DRAM",
    "DDR5": "Fifth generation DDR SDRAM with higher bandwidth and efficiency",
    "ROM": "Read-Only Memory — non-volatile memory with fixed data",
    "EEPROM": "Electrically Erasable Programmable ROM",
    "Flash Memory": "Non-volatile solid-state storage using floating-gate transistors",
    "NAND Flash": "Flash memory using NAND gates, basis of SSDs and USB drives",
    "NOR Flash": "Flash memory using NOR gates, used for firmware storage",
    "HDD": "Hard Disk Drive — magnetic spinning disk storage",
    "SSD": "Solid State Drive — flash-based storage with no moving parts",
    "NVMe": "Non-Volatile Memory Express — high-speed SSD interface over PCIe",
    "eMMC": "Embedded MultiMediaCard — flash storage for mobile devices",
    "Memory Controller": "Component managing data flow between CPU and memory",
    "Memory Bus": "Communication pathway between memory and CPU",
    "Virtual Memory": "Abstraction giving each process its own address space using disk backing",
    "Page Table": "Data structure mapping virtual to physical memory addresses",
    "TLB": "Translation Lookaside Buffer — cache for page table entries",
    "Memory Management Unit": "Hardware translating virtual addresses to physical addresses",
    "ECC Memory": "Error-Correcting Code memory detecting and correcting single-bit errors",
    # GPU
    "GPU": "Graphics Processing Unit — massively parallel processor for graphics and compute",
    "Shader Core": "Programmable GPU unit executing shader programs",
    "Vertex Shader": "GPU shader processing each vertex of 3D geometry",
    "Fragment Shader": "GPU shader computing color for each pixel fragment",
    "Compute Shader": "GPU shader for general-purpose parallel computation",
    "VRAM": "Video RAM — dedicated high-bandwidth memory for GPU",
    "GDDR6": "Graphics DDR6 — high-bandwidth memory for GPUs",
    "HBM": "High Bandwidth Memory — 3D-stacked DRAM for GPUs and accelerators",
    "Tensor Core": "NVIDIA GPU unit for accelerating matrix multiply-accumulate operations",
    "Ray Tracing Core": "GPU hardware unit accelerating ray-scene intersection calculations",
    "Rasterizer": "Hardware converting vector graphics to pixels for display",
    "Frame Buffer": "Memory buffer holding pixel data for display output",
    # Motherboard & buses
    "Motherboard": "Main circuit board connecting CPU, memory, storage, and peripherals",
    "Chipset": "Set of chips managing data flow between CPU, memory, and peripherals",
    "Northbridge": "Legacy chipset component connecting CPU to high-speed buses",
    "Southbridge": "Legacy chipset component connecting to lower-speed peripherals",
    "PCIe": "Peripheral Component Interconnect Express — high-speed serial expansion bus",
    "USB": "Universal Serial Bus — standard for connecting peripherals",
    "USB-C": "Reversible USB connector supporting high-speed data and power delivery",
    "Thunderbolt": "High-speed I/O interface combining PCIe and DisplayPort",
    "SATA": "Serial ATA — interface for connecting storage drives",
    "M.2": "Small form factor connector for SSDs and wireless cards",
    "BIOS": "Basic Input/Output System — legacy firmware initializing hardware at boot",
    "UEFI": "Unified Extensible Firmware Interface — modern BIOS replacement",
    "CMOS": "Complementary Metal-Oxide-Semiconductor — stores BIOS settings",
    "POST": "Power-On Self-Test — hardware diagnostic run at startup",
    "DMA": "Direct Memory Access — hardware accessing memory without CPU intervention",
    "Interrupt": "Signal to CPU requesting immediate attention from a device",
    "IRQ": "Interrupt Request — hardware interrupt signal line",
    "I/O Port": "Interface for CPU to communicate with peripheral devices",
    "Bus": "Communication system transferring data between computer components",
    "System Bus": "Primary bus connecting CPU to main memory and I/O",
    "Address Bus": "Bus carrying memory addresses from CPU to memory",
    "Data Bus": "Bus carrying data between CPU, memory, and peripherals",
    # Power & cooling
    "PSU": "Power Supply Unit — converts AC to DC power for computer components",
    "VRM": "Voltage Regulator Module — regulates voltage for CPU and GPU",
    "Heat Sink": "Passive cooling component dissipating heat via thermal conduction",
    "CPU Fan": "Active cooling fan mounted on CPU heat sink",
    "Liquid Cooling": "Cooling system using liquid coolant circulated through radiator",
    "Thermal Paste": "Thermally conductive compound between CPU and heat sink",
    "TDP": "Thermal Design Power — maximum heat dissipation specification in watts",
    # Storage
    "RAID": "Redundant Array of Independent Disks — combining drives for performance or redundancy",
    "RAID 0": "Striping across drives for performance, no redundancy",
    "RAID 1": "Mirroring across drives for redundancy",
    "RAID 5": "Striping with distributed parity across 3+ drives",
    "RAID 10": "Combination of mirroring and striping for performance and redundancy",
    "NAS": "Network Attached Storage — file-level storage accessible over network",
    "SAN": "Storage Area Network — block-level high-speed storage network",
    "Tape Drive": "Sequential access storage using magnetic tape for archival",
    # Networking HW
    "NIC": "Network Interface Card — hardware connecting computer to network",
    "Ethernet Port": "RJ-45 connector for wired Ethernet network connections",
    "Wi-Fi Adapter": "Wireless network interface for IEEE 802.11 connections",
    "Switch": "Network device forwarding frames based on MAC addresses",
    "Router": "Network device forwarding packets between different networks",
    "Hub": "Legacy network device broadcasting all traffic to all ports",
    "Modem": "Device modulating/demodulating signals for network communication",
    "Firewall Hardware": "Dedicated device filtering network traffic for security",
    "Load Balancer Hardware": "Device distributing network traffic across multiple servers",
    "Access Point": "Device creating wireless local area network",
    "Fiber Optic Cable": "Cable transmitting data as light pulses through glass fibers",
    "Ethernet Cable": "Twisted pair copper cable for Ethernet connections (Cat5e/6/6a)",
    "Coaxial Cable": "Cable with central conductor for broadband and cable TV",
    # Display
    "Monitor": "Display device showing visual output from a computer",
    "LCD": "Liquid Crystal Display — flat panel using liquid crystals",
    "OLED": "Organic LED display with self-emitting pixels for high contrast",
    "LED Display": "Display using light-emitting diodes for backlighting or direct view",
    "Refresh Rate": "How many times per second a display updates, measured in Hz",
    "Resolution": "Number of pixels in display dimensions (e.g., 1920×1080)",
    "HDMI": "High-Definition Multimedia Interface for audio/video transmission",
    "DisplayPort": "Digital display interface for high-resolution video output",
    "VGA": "Video Graphics Array — legacy analog video connector",
    # Input
    "Keyboard": "Input device with keys for typing text and commands",
    "Mouse": "Pointing device for cursor control and selection",
    "Touchpad": "Touch-sensitive surface for cursor control on laptops",
    "Touchscreen": "Display that detects touch input directly on screen",
    "Microphone": "Device converting sound waves to electrical signals",
    "Webcam": "Camera for capturing video for communication or recording",
    "Scanner": "Device digitizing physical documents and images",
    "Stylus": "Pen-like input device for precise drawing on screens or tablets",
    # Other
    "TPU": "Tensor Processing Unit — Google's custom ASIC for ML acceleration",
    "NPU": "Neural Processing Unit — dedicated hardware for AI inference",
    "DPU": "Data Processing Unit — programmable processor for data-centric tasks",
    "Quantum Computer": "Computer using quantum mechanical phenomena for computation",
    "Qubit": "Quantum bit — unit of quantum information in superposition of 0 and 1",
    "Quantum Gate": "Basic operation on qubits analogous to classical logic gates",
    "3D Printer": "Device creating physical objects layer by layer from digital models",
    "Raspberry Pi": "Low-cost single-board computer for education and prototyping",
    "Arduino": "Open-source microcontroller platform for electronics prototyping",
    "Oscilloscope": "Electronic instrument displaying electrical signal waveforms",
    "Multimeter": "Instrument measuring voltage, current, and resistance",
    "Power Delivery": "USB-C specification for delivering up to 240W of power",
    "Bluetooth": "Short-range wireless technology for exchanging data between devices",
    "Zigbee": "Low-power wireless mesh network protocol for IoT devices",
    "Z-Wave": "Low-energy wireless protocol for home automation",
    "LoRa": "Long-range low-power wireless protocol for IoT applications",
    "5G": "Fifth generation mobile network technology with high speed and low latency",
    "LTE": "Long-Term Evolution — 4G wireless broadband standard",
    "Wi-Fi 6": "IEEE 802.11ax — latest Wi-Fi standard with improved throughput and efficiency",
    "Optical Drive": "Device reading/writing optical discs (CD, DVD, Blu-ray)",
    "Sound Card": "Hardware component processing audio input and output",
    "DAC": "Digital-to-Analog Converter — converts digital audio to analog signals",
    "ADC": "Analog-to-Digital Converter — converts analog signals to digital data",
    "FPGA Development Board": "Board with FPGA chip for prototyping digital circuits",
    "Server Rack": "Standardized frame for mounting server and networking equipment",
    "UPS": "Uninterruptible Power Supply — provides backup power during outages",
    "KVM Switch": "Device allowing one keyboard/video/mouse to control multiple computers",
}

C("Computer Hardware", "Physical components making up a computer system", "DEFINITION")
for name, defn in hw.items():
    C(name, defn, "FACT")
    R(name, "Computer Hardware", "IS_A")

# HW relations
R("ALU", "CPU", "PART_OF"); R("Control Unit", "CPU", "PART_OF")
R("Register", "CPU", "PART_OF"); R("Cache", "CPU", "PART_OF")
R("L1 Cache", "Cache", "IS_A"); R("L2 Cache", "Cache", "IS_A"); R("L3 Cache", "Cache", "IS_A")
R("Pipeline", "CPU", "PART_OF"); R("Branch Predictor", "CPU", "PART_OF")
R("DRAM", "RAM", "IS_A"); R("SRAM", "RAM", "IS_A")
R("DDR4", "DRAM", "IS_A"); R("DDR5", "DRAM", "IS_A")
R("NAND Flash", "Flash Memory", "IS_A"); R("NOR Flash", "Flash Memory", "IS_A")
R("SSD", "NAND Flash", "USED_IN"); R("NVMe", "SSD", "IS_A")
R("HDD", "SATA", "USED_IN"); R("SSD", "SATA", "USED_IN")
R("Shader Core", "GPU", "PART_OF"); R("VRAM", "GPU", "PART_OF")
R("Tensor Core", "GPU", "PART_OF"); R("Ray Tracing Core", "GPU", "PART_OF")
R("GDDR6", "VRAM", "IS_A"); R("HBM", "VRAM", "IS_A")
R("Chipset", "Motherboard", "PART_OF"); R("PCIe", "Motherboard", "PART_OF")
R("VRM", "Motherboard", "PART_OF")
R("UEFI", "BIOS", "RELATES_TO")
R("CPU", "Motherboard", "PART_OF"); R("RAM", "Motherboard", "PART_OF")
R("GPU", "PCIe", "USED_IN")
R("Transistor", "MOSFET", "IS_A")
R("MOSFET", "Logic Gate", "USED_IN")
R("AND Gate", "Logic Gate", "IS_A"); R("OR Gate", "Logic Gate", "IS_A")
R("NOT Gate", "Logic Gate", "IS_A"); R("NAND Gate", "Logic Gate", "IS_A")
R("NOR Gate", "Logic Gate", "IS_A"); R("XOR Gate", "Logic Gate", "IS_A")
R("Flip-Flop", "SRAM", "USED_IN")
R("Full Adder", "Adder", "IS_A"); R("Half Adder", "Adder", "IS_A")
R("RAID 0", "RAID", "IS_A"); R("RAID 1", "RAID", "IS_A")
R("RAID 5", "RAID", "IS_A"); R("RAID 10", "RAID", "IS_A")
R("TPU", "Neural Network", "ENABLES"); R("NPU", "Neural Network", "ENABLES")
R("FPGA", "VHDL", "USED_IN"); R("FPGA", "Verilog", "USED_IN")
R("Arduino", "Microcontroller", "IS_A")
R("Qubit", "Quantum Computer", "PART_OF")
R("Quantum Gate", "Quantum Computer", "PART_OF")
R("ARM Architecture", "RISC", "IS_A")
R("x86 Architecture", "CISC", "IS_A")
R("RISC-V", "RISC", "IS_A")

# ============================================================
# NETWORKING PROTOCOLS (110+)
# ============================================================
protos = {
    # Application layer
    "HTTP": "HyperText Transfer Protocol — request-response protocol for web communication",
    "HTTPS": "HTTP over TLS — encrypted secure web communication",
    "HTTP/2": "Major revision of HTTP with multiplexing and header compression",
    "HTTP/3": "HTTP over QUIC — UDP-based transport with built-in encryption",
    "FTP": "File Transfer Protocol — transferring files between client and server",
    "SFTP": "SSH File Transfer Protocol — secure file transfer over SSH",
    "FTPS": "FTP over TLS — encrypted file transfer",
    "SSH": "Secure Shell — encrypted remote command-line access protocol",
    "Telnet": "Legacy unencrypted remote terminal protocol",
    "SMTP": "Simple Mail Transfer Protocol — sending email between servers",
    "IMAP": "Internet Message Access Protocol — accessing email on remote server",
    "POP3": "Post Office Protocol v3 — downloading email from server",
    "DNS": "Domain Name System — resolving domain names to IP addresses",
    "DHCP": "Dynamic Host Configuration Protocol — automatic IP address assignment",
    "SNMP": "Simple Network Management Protocol — monitoring network devices",
    "NTP": "Network Time Protocol — synchronizing clocks across networks",
    "LDAP": "Lightweight Directory Access Protocol — accessing directory services",
    "XMPP": "Extensible Messaging and Presence Protocol — real-time messaging",
    "MQTT": "Message Queuing Telemetry Transport — lightweight IoT messaging protocol",
    "CoAP": "Constrained Application Protocol — lightweight protocol for IoT devices",
    "AMQP": "Advanced Message Queuing Protocol — message-oriented middleware",
    "SIP": "Session Initiation Protocol — signaling for voice and video calls",
    "RTP": "Real-time Transport Protocol — delivering audio and video over IP",
    "RTSP": "Real Time Streaming Protocol — controlling streaming media servers",
    "WebRTC": "Web Real-Time Communication — peer-to-peer audio/video/data in browsers",
    "SOAP": "Simple Object Access Protocol — XML-based messaging for web services",
    "gRPC Protocol": "Google's high-performance RPC protocol using HTTP/2 and Protocol Buffers",
    "BitTorrent": "Peer-to-peer file sharing protocol using distributed hash table",
    "NFS": "Network File System — distributed file system protocol for remote file access",
    "SMB": "Server Message Block — network file sharing protocol used by Windows",
    "CIFS": "Common Internet File System — dialect of SMB for Windows file sharing",
    "iSCSI": "Internet Small Computer Systems Interface — block storage over IP networks",
    "Kerberos": "Network authentication protocol using tickets for secure identity verification",
    "RADIUS": "Remote Authentication Dial-In User Service — centralized authentication",
    "OAuth 2.0": "Authorization framework enabling third-party access to resources",
    "OpenID Connect": "Identity layer on top of OAuth 2.0 for authentication",
    "SAML": "Security Assertion Markup Language — XML-based SSO authentication",
    "WS-Security": "SOAP message security extension for web services",
    # Transport layer
    "TCP": "Transmission Control Protocol — reliable ordered byte stream delivery",
    "UDP": "User Datagram Protocol — connectionless unreliable fast datagram delivery",
    "QUIC": "UDP-based transport protocol with built-in TLS, used by HTTP/3",
    "SCTP": "Stream Control Transmission Protocol — multi-stream reliable transport",
    "TLS": "Transport Layer Security — cryptographic protocol for secure communication",
    "SSL": "Secure Sockets Layer — predecessor to TLS, now deprecated",
    "DTLS": "Datagram Transport Layer Security — TLS for datagram protocols",
    # Network layer
    "IP": "Internet Protocol — addressing and routing packets across networks",
    "IPv4": "Internet Protocol version 4 with 32-bit addresses",
    "IPv6": "Internet Protocol version 6 with 128-bit addresses",
    "ICMP": "Internet Control Message Protocol — error reporting and diagnostics (ping)",
    "ICMPv6": "ICMP for IPv6 including neighbor discovery",
    "IGMP": "Internet Group Management Protocol — managing multicast group membership",
    "IPsec": "Internet Protocol Security — encrypting and authenticating IP packets",
    "GRE": "Generic Routing Encapsulation — tunneling protocol encapsulating packets",
    "VXLAN": "Virtual Extensible LAN — network virtualization overlay over UDP",
    "MPLS": "Multi-Protocol Label Switching — high-performance traffic routing using labels",
    "BGP": "Border Gateway Protocol — routing between autonomous systems on the internet",
    "OSPF": "Open Shortest Path First — link-state routing for interior networks",
    "RIP": "Routing Information Protocol — distance-vector routing using hop count",
    "EIGRP": "Enhanced Interior Gateway Routing Protocol — Cisco's advanced distance-vector routing",
    "IS-IS": "Intermediate System to Intermediate System — link-state routing protocol",
    "ARP": "Address Resolution Protocol — mapping IP addresses to MAC addresses",
    "NDP": "Neighbor Discovery Protocol — IPv6 address resolution and router discovery",
    "NAT": "Network Address Translation — mapping private to public IP addresses",
    "VRRP": "Virtual Router Redundancy Protocol — automatic router failover",
    "HSRP": "Hot Standby Router Protocol — Cisco's gateway redundancy protocol",
    # Data link
    "Ethernet Protocol": "IEEE 802.3 standard for wired LAN communication",
    "Wi-Fi Protocol": "IEEE 802.11 standard for wireless LAN communication",
    "PPP": "Point-to-Point Protocol — data link for direct connections",
    "HDLC": "High-Level Data Link Control — synchronous data link protocol",
    "Frame Relay": "WAN protocol for efficient data transmission using virtual circuits",
    "ATM": "Asynchronous Transfer Mode — cell-based switching for voice and data",
    "STP": "Spanning Tree Protocol — preventing loops in Ethernet networks",
    "RSTP": "Rapid Spanning Tree Protocol — faster convergence variant of STP",
    "LACP": "Link Aggregation Control Protocol — bundling multiple links for bandwidth",
    "LLDP": "Link Layer Discovery Protocol — network device discovery",
    "CDP": "Cisco Discovery Protocol — proprietary network device discovery",
    "802.1Q": "IEEE standard for VLAN tagging on Ethernet frames",
    "802.1X": "IEEE port-based network access control standard",
    # Application protocols (additional)
    "TFTP": "Trivial File Transfer Protocol — simple UDP-based file transfer",
    "Syslog": "Standard for message logging across network devices",
    "PGP": "Pretty Good Privacy — encryption for email and file security",
    "S/MIME": "Secure MIME — email encryption and digital signing",
    "DNSSEC": "DNS Security Extensions — authenticating DNS responses",
    "DoH": "DNS over HTTPS — encrypted DNS queries via HTTPS",
    "DoT": "DNS over TLS — encrypted DNS queries via TLS",
    "WireGuard": "Modern VPN protocol with simple, fast, and secure design",
    "OpenVPN": "Open-source VPN solution using TLS for key exchange",
    "L2TP": "Layer 2 Tunneling Protocol — VPN tunneling often combined with IPsec",
    "PPTP": "Point-to-Point Tunneling Protocol — legacy VPN protocol",
    "TACACS+": "Terminal Access Controller Access-Control System Plus — AAA protocol",
    "DIAMETER": "Authentication protocol replacing RADIUS with improved capabilities",
    "ZeroConf": "Zero Configuration Networking — automatic network setup without manual config",
    "mDNS": "Multicast DNS — local network name resolution without a DNS server",
    "UPnP": "Universal Plug and Play — automatic device discovery and configuration",
    "DLNA": "Digital Living Network Alliance — media sharing protocol between devices",
}

C("Network Protocol", "Set of rules governing communication between network devices", "DEFINITION")
C("OSI Model", "Seven-layer conceptual model for network communication", "DEFINITION")
C("TCP/IP Model", "Four-layer practical model for internet communication", "DEFINITION")

for name, defn in protos.items():
    C(name, defn)
    R(name, "Network Protocol", "IS_A")

# Protocol layer relations
R("TCP", "IP", "USED_IN"); R("UDP", "IP", "USED_IN")
R("HTTP", "TCP", "USED_IN"); R("HTTPS", "TLS", "USED_IN"); R("HTTPS", "HTTP", "RELATES_TO")
R("HTTP/2", "TCP", "USED_IN"); R("HTTP/3", "QUIC", "USED_IN"); R("QUIC", "UDP", "USED_IN")
R("TLS", "TCP", "USED_IN"); R("SSH", "TCP", "USED_IN")
R("DNS", "UDP", "USED_IN"); R("DHCP", "UDP", "USED_IN")
R("FTP", "TCP", "USED_IN"); R("SMTP", "TCP", "USED_IN")
R("IPv4", "IP", "IS_A"); R("IPv6", "IP", "IS_A")
R("BGP", "TCP", "USED_IN"); R("OSPF", "IP", "USED_IN")
R("WireGuard", "UDP", "USED_IN")
R("DNSSEC", "DNS", "RELATES_TO"); R("DoH", "DNS", "RELATES_TO")
R("WebRTC", "RTP", "USED_IN"); R("WebRTC", "UDP", "USED_IN")
R("MQTT", "TCP", "USED_IN")
R("OAuth 2.0", "HTTPS", "USED_IN")
R("OpenID Connect", "OAuth 2.0", "RELATES_TO")

# ============================================================
# MATH THEOREMS & FORMULAS (200+)
# ============================================================
C("Mathematics", "Abstract science of number, quantity, and space", "DEFINITION")
C("Theorem", "Statement proven true from axioms and previously established theorems", "DEFINITION")
C("Proof", "Logical argument establishing the truth of a mathematical statement", "DEFINITION")
C("Axiom", "Self-evident truth accepted without proof as a basis for reasoning", "DEFINITION")
C("Conjecture", "Unproven mathematical statement believed to be true", "DEFINITION")

math_concepts = {
    # Algebra
    "Algebra": "Branch of mathematics dealing with symbols and rules for manipulating them",
    "Linear Algebra": "Study of vectors, vector spaces, linear mappings, and systems of equations",
    "Abstract Algebra": "Study of algebraic structures like groups, rings, and fields",
    "Group": "Set with binary operation satisfying closure, associativity, identity, and inverse",
    "Ring": "Set with two operations (addition, multiplication) forming an abelian group and monoid",
    "Field": "Ring where every nonzero element has a multiplicative inverse",
    "Vector Space": "Set of vectors closed under addition and scalar multiplication over a field",
    "Eigenvalue": "Scalar λ such that Av = λv for matrix A and nonzero vector v",
    "Eigenvector": "Nonzero vector unchanged in direction by a linear transformation",
    "Determinant": "Scalar value encoding properties of a square matrix, det(A)",
    "Matrix Multiplication": "Binary operation producing matrix from two matrices, rows × columns",
    "Inverse Matrix": "Matrix A⁻¹ such that AA⁻¹ = I, exists iff det(A) ≠ 0",
    "Rank": "Dimension of the column space (or row space) of a matrix",
    "Null Space": "Set of all vectors v where Av = 0 for matrix A",
    "Orthogonality": "Property of vectors being perpendicular (dot product = 0)",
    "Gram-Schmidt Process": "Algorithm for orthogonalizing a set of vectors in an inner product space",
    "Cayley-Hamilton Theorem": "Every square matrix satisfies its own characteristic equation",
    "Spectral Theorem": "Symmetric matrices have orthonormal eigenvectors with real eigenvalues",
    "Jordan Normal Form": "Canonical form showing eigenvalue structure of a matrix",
    "Polynomial": "Expression of variables and coefficients using addition, multiplication, and exponentiation",
    "Quadratic Formula": "x = (-b ± √(b²-4ac)) / 2a — roots of ax² + bx + c = 0",
    "Fundamental Theorem of Algebra": "Every non-constant polynomial has at least one complex root",
    "Binomial Theorem": "(x+y)ⁿ = Σ C(n,k) xⁿ⁻ᵏ yᵏ — expansion of binomial powers",
    "Vieta's Formulas": "Relations between polynomial roots and coefficients",
    "Galois Theory": "Study of field extensions and symmetries of polynomial roots",
    "Isomorphism": "Structure-preserving bijective mapping between algebraic structures",
    "Homomorphism": "Structure-preserving mapping between algebraic structures",
    # Calculus
    "Calculus": "Study of continuous change through derivatives and integrals",
    "Derivative": "Rate of change of a function, f'(x) = lim (f(x+h)-f(x))/h as h→0",
    "Integral": "Area under a curve, accumulation of quantities, antiderivative",
    "Fundamental Theorem of Calculus": "Differentiation and integration are inverse operations",
    "Chain Rule": "Derivative of composition: (f∘g)'(x) = f'(g(x))·g'(x)",
    "Product Rule": "Derivative of product: (fg)' = f'g + fg'",
    "Quotient Rule": "Derivative of quotient: (f/g)' = (f'g - fg')/g²",
    "L'Hôpital's Rule": "Limit of indeterminate form 0/0 or ∞/∞ equals limit of derivatives ratio",
    "Integration by Parts": "∫u dv = uv - ∫v du — integration technique from product rule",
    "Integration by Substitution": "∫f(g(x))g'(x)dx = ∫f(u)du — changing variable of integration",
    "Taylor Series": "f(x) = Σ f⁽ⁿ⁾(a)(x-a)ⁿ/n! — infinite polynomial approximation of a function",
    "Maclaurin Series": "Taylor series centered at a=0",
    "Power Series": "Infinite series Σ aₙxⁿ representing functions within radius of convergence",
    "Fourier Series": "Representing periodic function as sum of sines and cosines",
    "Fourier Transform": "Decomposing function into frequency components, continuous analog of Fourier series",
    "Laplace Transform": "Integral transform converting time-domain to s-domain: L{f(t)} = ∫f(t)e⁻ˢᵗdt",
    "Mean Value Theorem": "If f continuous on [a,b] and differentiable on (a,b), ∃c: f'(c)=(f(b)-f(a))/(b-a)",
    "Intermediate Value Theorem": "Continuous function on [a,b] takes every value between f(a) and f(b)",
    "Extreme Value Theorem": "Continuous function on closed interval attains maximum and minimum",
    "Rolle's Theorem": "If f(a)=f(b) and f continuous/differentiable, ∃c∈(a,b): f'(c)=0",
    "Squeeze Theorem": "If g(x)≤f(x)≤h(x) and lim g=lim h=L, then lim f=L",
    "Divergence Theorem": "Volume integral of divergence equals surface integral of flux",
    "Stokes' Theorem": "Integral of curl over surface equals line integral around boundary",
    "Green's Theorem": "Relates line integral around curve to double integral over enclosed region",
    "Partial Derivative": "Derivative with respect to one variable holding others constant",
    "Gradient": "Vector of partial derivatives pointing in direction of steepest ascent",
    "Jacobian": "Matrix of first-order partial derivatives of a vector-valued function",
    "Hessian": "Matrix of second-order partial derivatives for analyzing curvature",
    "Lagrange Multipliers": "Method for finding constrained extrema using auxiliary variables",
    "Differential Equation": "Equation involving derivatives of unknown functions",
    "Ordinary Differential Equation": "Differential equation with one independent variable",
    "Partial Differential Equation": "Differential equation with multiple independent variables",
    "Euler's Method": "First-order numerical method for solving ODEs by linear steps",
    "Runge-Kutta Method": "Family of numerical ODE solvers using weighted average of slopes",
    # Number theory
    "Number Theory": "Study of integers and integer-valued functions",
    "Prime Number": "Integer greater than 1 divisible only by 1 and itself",
    "Fundamental Theorem of Arithmetic": "Every integer > 1 is uniquely factorable into primes",
    "Euclid's Theorem": "There are infinitely many prime numbers",
    "Fermat's Little Theorem": "aᵖ ≡ a (mod p) for prime p — basis of primality testing",
    "Euler's Theorem": "aᵠ⁽ⁿ⁾ ≡ 1 (mod n) for gcd(a,n)=1, generalizes Fermat's little theorem",
    "Chinese Remainder Theorem": "System of congruences with coprime moduli has unique solution",
    "Goldbach's Conjecture": "Every even integer > 2 is the sum of two primes (unproven)",
    "Twin Prime Conjecture": "There are infinitely many primes p where p+2 is also prime (unproven)",
    "Riemann Hypothesis": "All nontrivial zeros of ζ(s) have real part 1/2 (unproven millennium problem)",
    "Prime Number Theorem": "Number of primes ≤ n is approximately n/ln(n)",
    "Modular Arithmetic": "Arithmetic system where numbers wrap around after reaching modulus",
    "Euler's Totient Function": "φ(n) counts integers 1 to n coprime to n",
    "Diophantine Equation": "Polynomial equation seeking integer solutions",
    "Fermat's Last Theorem": "xⁿ + yⁿ = zⁿ has no positive integer solutions for n > 2 (proven by Wiles)",
    "Quadratic Reciprocity": "Law relating solvability of quadratic equations in modular arithmetic",
    "Bézout's Identity": "For integers a,b: ∃x,y such that ax + by = gcd(a,b)",
    "Wilson's Theorem": "(p-1)! ≡ -1 (mod p) if and only if p is prime",
    "Legendre's Formula": "Exact power of prime p dividing n! using floor sums",
    # Combinatorics
    "Combinatorics": "Study of counting, arrangement, and combination of objects",
    "Permutation": "Ordered arrangement of r items from n: P(n,r) = n!/(n-r)!",
    "Combination": "Unordered selection of r items from n: C(n,r) = n!/(r!(n-r)!)",
    "Pigeonhole Principle": "If n items placed in m containers (n>m), at least one has >1 item",
    "Inclusion-Exclusion Principle": "|A∪B| = |A| + |B| - |A∩B|, generalized to n sets",
    "Catalan Numbers": "Sequence counting balanced parentheses, binary trees, etc: Cₙ = C(2n,n)/(n+1)",
    "Fibonacci Sequence": "Sequence where each number is sum of two preceding: 0, 1, 1, 2, 3, 5, 8...",
    "Pascal's Triangle": "Triangular array of binomial coefficients with each entry sum of two above",
    "Stirling Numbers": "Count partitions of n elements into k non-empty subsets",
    "Bell Numbers": "Count total number of partitions of a set",
    "Burnside's Lemma": "Counting distinct objects under group symmetry using fixed points",
    "Ramsey Theory": "Study of conditions guaranteeing order within large structures",
    "Generating Function": "Formal power series encoding a sequence for combinatorial analysis",
    "Recurrence Relation": "Equation defining sequence terms based on previous terms",
    "Master Theorem": "Formula for solving divide-and-conquer recurrences T(n) = aT(n/b) + f(n)",
    # Geometry
    "Geometry": "Study of shapes, sizes, positions, and properties of space",
    "Euclidean Geometry": "Geometry based on Euclid's axioms in flat space",
    "Non-Euclidean Geometry": "Geometry where parallel postulate does not hold",
    "Pythagorean Theorem": "In right triangle: a² + b² = c² where c is hypotenuse",
    "Law of Cosines": "c² = a² + b² - 2ab·cos(C) — generalization of Pythagorean theorem",
    "Law of Sines": "a/sin(A) = b/sin(B) = c/sin(C) — relating sides to angles",
    "Triangle Inequality": "Sum of any two sides of a triangle exceeds the third side",
    "Euler's Formula (Polyhedra)": "V - E + F = 2 for convex polyhedra (vertices, edges, faces)",
    "Area of Circle": "A = πr² where r is radius",
    "Circumference": "C = 2πr — distance around a circle",
    "Volume of Sphere": "V = (4/3)πr³",
    "Surface Area of Sphere": "A = 4πr²",
    "Thales' Theorem": "Angle inscribed in semicircle is always 90 degrees",
    "Ceva's Theorem": "Concurrency condition for cevians in a triangle",
    "Menelaus' Theorem": "Collinearity condition for points on sides of a triangle",
    "Stewart's Theorem": "Relating lengths of cevian, sides, and segments of a triangle",
    "Apollonius' Theorem": "Median length formula: 2(a²+b²) = 4m² + c²",
    "Affine Transformation": "Linear mapping preserving points, lines, and parallelism",
    "Projective Geometry": "Geometry studying properties invariant under projection",
    "Hyperbolic Geometry": "Non-Euclidean geometry with negative curvature",
    "Elliptic Geometry": "Non-Euclidean geometry with positive curvature (sphere surface)",
    "Fractal": "Self-similar geometric shape at every scale",
    "Mandelbrot Set": "Fractal defined by iteration of zₙ₊₁ = zₙ² + c in complex plane",
    # Topology
    "Topology": "Study of properties preserved under continuous deformations",
    "Homeomorphism": "Continuous bijection with continuous inverse — topological equivalence",
    "Compact Space": "Topological space where every open cover has a finite subcover",
    "Connected Space": "Topological space that cannot be divided into two disjoint open sets",
    "Hausdorff Space": "Topological space where distinct points have disjoint neighborhoods",
    "Euler Characteristic": "Topological invariant χ = V - E + F",
    "Fixed Point Theorem": "Brouwer's theorem: continuous map from disk to itself has a fixed point",
    # Probability & Statistics
    "Probability Theory": "Mathematical study of random phenomena and likelihood",
    "Bayes' Theorem": "P(A|B) = P(B|A)P(A)/P(B) — updating probability with evidence",
    "Law of Large Numbers": "Sample average converges to expected value as sample size increases",
    "Central Limit Theorem": "Sum of independent random variables tends toward normal distribution",
    "Normal Distribution": "Bell-curve distribution with mean μ and standard deviation σ",
    "Poisson Distribution": "Distribution of events occurring in fixed interval with constant rate",
    "Binomial Distribution": "Distribution of successes in n independent Bernoulli trials",
    "Exponential Distribution": "Distribution of time between events in a Poisson process",
    "Uniform Distribution": "Distribution where all outcomes are equally likely",
    "Markov Chain": "Stochastic process where future depends only on present state",
    "Expected Value": "Long-run average of a random variable: E[X] = Σ xP(x)",
    "Variance": "Measure of spread: Var(X) = E[(X-μ)²]",
    "Standard Deviation": "Square root of variance, same units as the data",
    "Covariance": "Measure of joint variability between two random variables",
    "Correlation": "Normalized covariance measuring linear relationship strength, [-1, 1]",
    "Conditional Probability": "P(A|B) — probability of A given B has occurred",
    "Independence": "Events A and B where P(A∩B) = P(A)P(B)",
    "Chi-Squared Test": "Statistical test comparing observed to expected frequencies",
    "T-Test": "Statistical test comparing means of two groups",
    "ANOVA": "Analysis of Variance — comparing means across multiple groups",
    "Regression Analysis": "Statistical method modeling relationship between variables",
    "Maximum Likelihood Estimation": "Estimating parameters by maximizing probability of observed data",
    "Hypothesis Testing": "Statistical procedure testing claims about population parameters",
    "P-Value": "Probability of observing result at least as extreme as test statistic under null hypothesis",
    "Confidence Interval": "Range of values likely containing the true population parameter",
    "Monte Carlo Method": "Computational technique using random sampling for numerical results",
    # Set Theory
    "Set Theory": "Study of collections of distinct objects and their properties",
    "Cantor's Theorem": "Power set of any set has strictly greater cardinality than the set",
    "Zorn's Lemma": "Every partially ordered set with upper bounds for chains has a maximal element",
    "Axiom of Choice": "For any collection of non-empty sets, a choice function exists",
    "Continuum Hypothesis": "No set has cardinality strictly between ℵ₀ and 2^ℵ₀ (independent of ZFC)",
    "Countable Set": "Set whose elements can be put in one-to-one correspondence with natural numbers",
    "Uncountable Set": "Set larger than the natural numbers (e.g., real numbers)",
    "Cardinality": "Measure of the number of elements in a set",
    "Power Set": "Set of all subsets of a given set, has cardinality 2ⁿ",
    "De Morgan's Laws": "Complement of union is intersection of complements, and vice versa",
    "Venn Diagram": "Diagram showing logical relationships between sets using overlapping circles",
    # Information Theory
    "Information Theory": "Mathematical study of quantifying and communicating information",
    "Shannon Entropy": "H = -Σ p(x) log p(x) — measure of information content",
    "Mutual Information": "Measure of mutual dependence between two random variables",
    "Kolmogorov Complexity": "Length of shortest program producing a given string",
    "Channel Capacity": "Maximum rate of reliable information transmission over a channel",
    "Huffman Coding": "Optimal prefix-free coding assigning shorter codes to frequent symbols",
    "Shannon's Source Coding Theorem": "Data can be compressed to at most entropy bits per symbol",
    "Error Correcting Code": "Code allowing detection and correction of transmission errors",
    "Hamming Code": "Linear error-correcting code detecting up to 2 and correcting 1 bit error",
    "Reed-Solomon Code": "Error-correcting code based on polynomial evaluation over finite fields",
    # Complexity Theory
    "Computational Complexity": "Study of resources needed to solve computational problems",
    "P Class": "Problems solvable in polynomial time by deterministic Turing machine",
    "NP Class": "Problems verifiable in polynomial time by nondeterministic Turing machine",
    "P vs NP Problem": "Open question whether P = NP — millennium prize problem",
    "NP-Complete": "Hardest problems in NP: if any is in P, then P = NP",
    "NP-Hard": "Problems at least as hard as NP-Complete, may not be in NP",
    "Big Omega Notation": "Asymptotic lower bound on growth rate of a function",
    "Big Theta Notation": "Tight asymptotic bound (both upper and lower)",
    "Turing Machine": "Abstract mathematical model of computation defining computability",
    "Halting Problem": "Undecidable problem: no algorithm can determine if any program halts",
    "Church-Turing Thesis": "Any effectively computable function is computable by a Turing machine",
    "Cook-Levin Theorem": "Boolean satisfiability (SAT) is NP-Complete",
    "Reduction": "Transforming one problem into another to prove complexity relationships",
    "Boolean Satisfiability": "Problem of determining if a Boolean formula can be satisfied",
    "3-SAT": "Satisfiability of Boolean formula in conjunctive normal form with 3 literals per clause",
    "Knapsack Problem": "NP-hard optimization: maximize value in capacity-limited knapsack",
    "Graph Isomorphism": "Determining if two graphs are structurally identical",
    "Space Complexity Class": "Classification of problems by memory required",
    "PSPACE": "Problems solvable with polynomial space",
    "EXPTIME": "Problems solvable in exponential time",
    "Automata Theory": "Study of abstract machines and problems they can solve",
    "Finite Automaton": "Machine with finite states processing input symbols",
    "Pushdown Automaton": "Finite automaton augmented with a stack",
    "Regular Language": "Language recognized by a finite automaton or regular expression",
    "Context-Free Grammar": "Grammar where production rules have single non-terminals on left",
    "Chomsky Hierarchy": "Classification of formal languages by generative power of grammars",
    # Misc important
    "Euler's Identity": "e^(iπ) + 1 = 0 — linking five fundamental mathematical constants",
    "Euler's Formula (Complex)": "e^(ix) = cos(x) + i·sin(x)",
    "Pi (π)": "Ratio of circle's circumference to diameter, approximately 3.14159",
    "Euler's Number (e)": "Base of natural logarithm, approximately 2.71828, limit of (1+1/n)ⁿ",
    "Imaginary Unit (i)": "Defined as √(-1), basis of complex numbers",
    "Complex Number": "Number of form a + bi where a, b are real and i² = -1",
    "Logarithm": "Inverse of exponentiation: log_b(x) = y iff bʸ = x",
    "Natural Logarithm": "Logarithm with base e: ln(x)",
    "Exponentiation": "Operation of raising a base to a power: aⁿ",
    "Factorial": "Product of all positive integers up to n: n! = 1·2·3·...·n",
    "Infinity": "Concept of unboundedness, not a number but a limit concept",
    "Continuum": "Real number line as a complete ordered field",
    "Limit": "Value a function approaches as input approaches a point",
    "Convergence": "Property of a sequence or series approaching a finite value",
    "Divergence": "Property of a sequence or series not approaching any finite value",
    "Completeness": "Property of a space where every Cauchy sequence converges",
    "Mathematical Induction": "Proof technique: prove base case, then inductive step implies all n",
    "Proof by Contradiction": "Assuming negation of statement and deriving a contradiction",
    "Proof by Contrapositive": "Proving ¬Q → ¬P to establish P → Q",
    "Pigeonhole Proof": "Existence proof using the pigeonhole principle",
    "Constructive Proof": "Proof that explicitly constructs the object claimed to exist",
    # Graph theory (mathematical)
    "Graph Theory": "Study of graphs as mathematical structures modeling pairwise relations",
    "Planar Graph": "Graph that can be drawn in the plane without edge crossings",
    "Complete Graph": "Graph where every pair of vertices is connected by an edge",
    "Bipartite Graph": "Graph whose vertices can be divided into two independent sets",
    "Tree (Graph Theory)": "Connected acyclic undirected graph with n-1 edges for n vertices",
    "Spanning Tree": "Subgraph that is a tree including all vertices of the graph",
    "Euler's Bridges Theorem": "Graph has Eulerian circuit iff every vertex has even degree",
    "Four Color Theorem": "Any planar map can be colored with at most 4 colors",
    "Hall's Marriage Theorem": "Bipartite graph has perfect matching iff Hall's condition holds",
    "Menger's Theorem": "Max number of disjoint paths = min vertex cut between two vertices",
    "Kuratowski's Theorem": "Graph is planar iff it contains no K₅ or K₃,₃ subdivision",
    "Chromatic Number": "Minimum colors needed to color graph vertices with no adjacent same color",
    "Clique": "Complete subgraph — subset of vertices all mutually adjacent",
    "Independent Set": "Set of vertices with no edges between them",
    "Hamiltonian Cycle": "Cycle visiting every vertex exactly once",
    "Eulerian Path": "Path traversing every edge exactly once",
    # Optimization
    "Optimization": "Finding the best solution from a set of feasible solutions",
    "Linear Programming": "Optimization of linear objective subject to linear constraints",
    "Integer Programming": "Linear programming where variables must be integers",
    "Convex Optimization": "Optimization of convex functions over convex sets — global optimum guaranteed",
    "Duality": "Every optimization problem has a dual providing bounds on the optimal value",
    "Strong Duality": "Primal and dual optimal values are equal (holds for convex problems)",
    "KKT Conditions": "Necessary conditions for optimality in constrained optimization",
    "Gradient": "Vector of partial derivatives indicating direction of steepest increase",
    "Heuristic": "Problem-solving approach that's not guaranteed optimal but practically useful",
    "NP-Hard Optimization": "Optimization problems that are NP-hard to solve exactly",
}

for name, defn in math_concepts.items():
    typ = "THEORY" if any(w in name for w in ["Theorem", "Conjecture", "Hypothesis", "Lemma", "Thesis"]) else "DEFINITION"
    C(name, defn, typ)

# Math branch relations
for c in ["Linear Algebra", "Abstract Algebra", "Galois Theory"]:
    R(c, "Algebra", "PART_OF")
for c in ["Derivative", "Integral", "Taylor Series", "Fourier Series", "Differential Equation"]:
    R(c, "Calculus", "PART_OF")
for c in ["Prime Number", "Fermat's Little Theorem", "Chinese Remainder Theorem", "Modular Arithmetic"]:
    R(c, "Number Theory", "PART_OF")
for c in ["Permutation", "Combination", "Catalan Numbers", "Fibonacci Sequence", "Generating Function"]:
    R(c, "Combinatorics", "PART_OF")
for c in ["Pythagorean Theorem", "Euclidean Geometry", "Non-Euclidean Geometry"]:
    R(c, "Geometry", "PART_OF")
for c in ["Bayes' Theorem", "Normal Distribution", "Markov Chain", "Central Limit Theorem"]:
    R(c, "Probability Theory", "PART_OF")
for c in ["Shannon Entropy", "Huffman Coding", "Channel Capacity"]:
    R(c, "Information Theory", "PART_OF")
for c in ["P Class", "NP Class", "Turing Machine", "Halting Problem"]:
    R(c, "Computational Complexity", "PART_OF")

R("Group", "Abstract Algebra", "PART_OF"); R("Ring", "Abstract Algebra", "PART_OF")
R("Field", "Abstract Algebra", "PART_OF"); R("Field", "Ring", "IS_A")
R("Ring", "Group", "RELATES_TO")
R("Vector Space", "Linear Algebra", "PART_OF")
R("Eigenvalue", "Linear Algebra", "PART_OF")
R("Eigenvector", "Eigenvalue", "RELATES_TO")
R("Determinant", "Matrix", "HAS_PROPERTY")
R("Fundamental Theorem of Calculus", "Derivative", "RELATES_TO")
R("Fundamental Theorem of Calculus", "Integral", "RELATES_TO")
R("Chain Rule", "Derivative", "RELATES_TO")
R("Taylor Series", "Power Series", "IS_A")
R("Maclaurin Series", "Taylor Series", "IS_A")
R("Fourier Transform", "Fourier Series", "RELATES_TO")
R("Fast Fourier Transform", "Fourier Transform", "ENABLES")
R("Laplace Transform", "Differential Equation", "USED_IN")
R("Euler's Method", "Ordinary Differential Equation", "USED_IN")
R("Runge-Kutta Method", "Ordinary Differential Equation", "USED_IN")
R("Ordinary Differential Equation", "Differential Equation", "IS_A")
R("Partial Differential Equation", "Differential Equation", "IS_A")
R("Stokes' Theorem", "Green's Theorem", "RELATES_TO")
R("Divergence Theorem", "Stokes' Theorem", "RELATES_TO")
R("Gradient Descent", "Gradient", "USED_IN")
R("Lagrange Multipliers", "Optimization", "USED_IN")
R("Linear Programming", "Optimization", "IS_A")
R("Integer Programming", "Linear Programming", "RELATES_TO")
R("Convex Optimization", "Optimization", "IS_A")
R("Simplex Algorithm", "Linear Programming", "USED_IN")
R("KKT Conditions", "Convex Optimization", "USED_IN")
R("NP-Complete", "NP Class", "PART_OF")
R("NP-Hard", "NP-Complete", "RELATES_TO")
R("Cook-Levin Theorem", "NP-Complete", "RELATES_TO")
R("Boolean Satisfiability", "NP-Complete", "IS_A")
R("3-SAT", "Boolean Satisfiability", "IS_A")
R("Knapsack Problem", "NP-Hard", "IS_A")
R("Traveling Salesman Problem", "NP-Hard", "IS_A")
R("Turing Machine", "Automata Theory", "PART_OF")
R("Finite Automaton", "Automata Theory", "PART_OF")
R("Pushdown Automaton", "Automata Theory", "PART_OF")
R("Regular Language", "Finite Automaton", "RELATES_TO")
R("Context-Free Grammar", "Pushdown Automaton", "RELATES_TO")
R("Chomsky Hierarchy", "Automata Theory", "RELATES_TO")
R("Halting Problem", "Turing Machine", "RELATES_TO")
R("Cantor's Theorem", "Set Theory", "PART_OF")
R("Axiom of Choice", "Set Theory", "PART_OF")
R("Mathematical Induction", "Proof", "IS_A")
R("Proof by Contradiction", "Proof", "IS_A")
R("Proof by Contrapositive", "Proof", "IS_A")
R("Complex Number", "Field", "IS_A")
R("Euler's Identity", "Euler's Formula (Complex)", "RELATES_TO")
R("Euler's Formula (Complex)", "Complex Number", "RELATES_TO")
R("Euler's Formula (Complex)", "Pi (π)", "RELATES_TO")
R("Euler's Formula (Complex)", "Euler's Number (e)", "RELATES_TO")
R("RSA Algorithm", "Fermat's Little Theorem", "USED_IN")
R("RSA Algorithm", "Euler's Theorem", "USED_IN")
R("Huffman Coding", "Greedy Algorithm", "USED_IN")
R("Huffman Coding", "Binary Tree", "USED_IN")
R("Master Theorem", "Recurrence Relation", "RELATES_TO")
R("Master Theorem", "Divide and Conquer", "RELATES_TO")
R("Fibonacci Sequence", "Recurrence Relation", "IS_A")
R("Four Color Theorem", "Planar Graph", "RELATES_TO")
R("Hamiltonian Cycle", "Traveling Salesman Problem", "RELATES_TO")
R("Spanning Tree", "Tree (Graph Theory)", "IS_A")
R("Bipartite Graph", "Graph Theory", "PART_OF")
R("Planar Graph", "Graph Theory", "PART_OF")

# ============================================================
# LOGIC (100+)
# ============================================================
logic = {
    "Logic": "Study of valid reasoning and argumentation",
    "Propositional Logic": "Logic dealing with propositions and logical connectives",
    "Predicate Logic": "Logic extending propositional logic with quantifiers and predicates",
    "First-Order Logic": "Predicate logic with quantifiers over individuals",
    "Second-Order Logic": "Logic with quantification over properties and relations",
    "Modal Logic": "Logic of necessity and possibility using □ and ◇ operators",
    "Temporal Logic": "Logic reasoning about propositions over time",
    "Fuzzy Logic": "Logic handling degrees of truth between 0 and 1",
    "Intuitionistic Logic": "Logic rejecting law of excluded middle, requiring constructive proofs",
    "Paraconsistent Logic": "Logic tolerating contradictions without triviality",
    "Deontic Logic": "Logic of obligation, permission, and prohibition",
    "Epistemic Logic": "Logic of knowledge and belief",
    "Boolean Algebra": "Algebraic structure capturing AND, OR, NOT operations on truth values",
    "Truth Table": "Table showing all possible truth values of a logical expression",
    "Tautology": "Statement that is true under every possible interpretation",
    "Contradiction": "Statement that is false under every possible interpretation",
    "Contingency": "Statement that is neither a tautology nor a contradiction",
    "Logical Conjunction": "AND operation: true only when both operands are true",
    "Logical Disjunction": "OR operation: true when at least one operand is true",
    "Logical Negation": "NOT operation: inverts the truth value",
    "Logical Implication": "If P then Q: false only when P is true and Q is false",
    "Logical Biconditional": "P if and only if Q: true when both have same truth value",
    "Exclusive Or": "XOR: true when exactly one operand is true",
    "Modus Ponens": "If P→Q and P, then Q — fundamental inference rule",
    "Modus Tollens": "If P→Q and ¬Q, then ¬P — contrapositive inference",
    "Hypothetical Syllogism": "If P→Q and Q→R, then P→R",
    "Disjunctive Syllogism": "If P∨Q and ¬P, then Q",
    "Constructive Dilemma": "If (P→Q)∧(R→S) and P∨R, then Q∨S",
    "Resolution": "Inference rule combining clauses by eliminating complementary literals",
    "Unification": "Finding substitution making logical expressions identical",
    "Skolemization": "Removing existential quantifiers by introducing Skolem functions",
    "Prenex Normal Form": "Logical formula with all quantifiers at the front",
    "Conjunctive Normal Form": "Conjunction of disjunctions of literals (AND of ORs)",
    "Disjunctive Normal Form": "Disjunction of conjunctions of literals (OR of ANDs)",
    "Universal Quantifier": "∀ — for all: statement holds for every element",
    "Existential Quantifier": "∃ — there exists: at least one element satisfies the statement",
    "Logical Equivalence": "Two statements having the same truth value in all interpretations",
    "Contrapositive": "Logically equivalent form: P→Q ≡ ¬Q→¬P",
    "Converse": "Reversing implication: converse of P→Q is Q→P",
    "Inverse": "Negating both parts: inverse of P→Q is ¬P→¬Q",
    "Soundness": "Property where all provable statements are true",
    "Completeness (Logic)": "Property where all true statements are provable",
    "Decidability": "Property of a logical system where validity is algorithmically determinable",
    "Gödel's Incompleteness Theorem": "Any consistent formal system powerful enough for arithmetic has unprovable true statements",
    "Gödel's Completeness Theorem": "Every valid first-order logic formula is provable",
    "Compactness Theorem": "Set of first-order sentences has a model iff every finite subset has",
    "Löwenheim-Skolem Theorem": "First-order theory with infinite model has models of every infinite cardinality",
    "Deduction Theorem": "If Γ∪{A} ⊢ B, then Γ ⊢ A→B",
    "Natural Deduction": "Proof system using introduction and elimination rules for each connective",
    "Sequent Calculus": "Proof system based on sequents (premises ⊢ conclusions)",
    "Hilbert System": "Axiom-rich proof system with few inference rules",
    "Proof Theory": "Study of proofs as formal mathematical objects",
    "Model Theory": "Study of relationships between formal languages and their interpretations",
    "Satisfiability": "Property of a formula being true under some interpretation",
    "Validity": "Property of a formula being true under all interpretations",
    "Entailment": "Relationship where truth of premises guarantees truth of conclusion",
    "Consistency": "Property of a set of formulas having no derivable contradiction",
    "Well-Formed Formula": "String of symbols that follows the syntax rules of a formal language",
    "Atomic Proposition": "Simplest statement with no logical connectives",
    "Literal": "Atomic proposition or its negation",
    "Clause": "Disjunction of literals in propositional logic",
    "Horn Clause": "Clause with at most one positive literal, basis of Prolog",
    "Definite Clause": "Horn clause with exactly one positive literal",
    "Empty Clause": "Clause with no literals, representing a contradiction",
    "Herbrand's Theorem": "First-order formula is unsatisfiable iff ground instances are unsatisfiable",
    "Robinson's Resolution": "Complete refutation procedure for first-order logic",
    "Craig's Interpolation Theorem": "If A⊢B, there exists interpolant C using only shared symbols",
    "Beth's Definability Theorem": "Implicit definability implies explicit definability in first-order logic",
    "Deductive Reasoning": "Drawing specific conclusions from general premises",
    "Inductive Reasoning": "Drawing general conclusions from specific observations",
    "Abductive Reasoning": "Inferring the best explanation for observations",
    "Formal Verification": "Mathematically proving correctness of systems and algorithms",
    "Model Checking": "Automatically verifying finite-state systems against specifications",
    "Theorem Proving": "Automated or assisted derivation of proofs for mathematical theorems",
    "SAT Solver": "Algorithm solving Boolean satisfiability problems",
    "SMT Solver": "Algorithm solving satisfiability modulo theories (SAT with background theories)",
    "BDD": "Binary Decision Diagram — compressed representation of Boolean functions",
    "Lambda Calculus": "Formal system for expressing computation via function abstraction and application",
    "Combinatory Logic": "Notation eliminating bound variables using combinators S, K, I",
    "Type Theory": "Formal system classifying expressions by types to prevent paradoxes",
    "Dependent Type": "Type depending on a value, enabling proofs as programs",
    "Curry-Howard Isomorphism": "Correspondence between proofs and programs, propositions and types",
    "Hoare Logic": "Formal system for reasoning about program correctness with pre/postconditions",
    "Separation Logic": "Extension of Hoare logic for reasoning about programs with mutable data structures",
    "Linear Logic": "Logic controlling resource usage — each assumption used exactly once",
    "Category Theory": "Abstract mathematical theory of structures and structure-preserving mappings",
    "Functor": "Mapping between categories preserving structure (objects and morphisms)",
    "Monad": "Abstraction in category theory used for sequencing computations in functional programming",
    "Fixed Point": "Value unchanged by a function: f(x) = x",
    "Lattice": "Partially ordered set where every pair has a least upper bound and greatest lower bound",
    "Boolean Lattice": "Complemented distributive lattice isomorphic to Boolean algebra",
    "Formal Language": "Set of strings over an alphabet defined by formal rules",
    "Regular Expression": "Pattern describing a set of strings using concatenation, union, and Kleene star",
    "Context-Free Language": "Language generated by a context-free grammar",
    "Turing Complete": "System capable of simulating any Turing machine",
    "Rice's Theorem": "All non-trivial semantic properties of programs are undecidable",
    "Post's Theorem": "Characterization of the arithmetical hierarchy using oracle machines",
    "Recursion Theory": "Study of computable functions and degrees of unsolvability",
    "Primitive Recursive Function": "Function computable using bounded recursion and composition",
    "Mu-Recursive Function": "Function computable using unbounded minimization, equivalent to Turing machines",
    "Church Encoding": "Representing data and operators in pure lambda calculus",
    "Y Combinator": "Fixed-point combinator enabling anonymous recursion in lambda calculus",
    "Curry's Paradox": "If self-reference is allowed, any statement can be proven true",
    "Russell's Paradox": "Set of all sets not containing themselves — contradiction in naive set theory",
    "Liar Paradox": "'This statement is false' — self-referential contradiction",
    "Zermelo-Fraenkel Set Theory": "Standard axiomatic set theory (ZF), with Choice becomes ZFC",
}

for name, defn in logic.items():
    typ = "THEORY" if any(w in name for w in ["Theorem", "Paradox", "Isomorphism"]) else "DEFINITION"
    C(name, defn, typ)
    R(name, "Logic", "PART_OF" if name != "Logic" else "IS_A")

R("Predicate Logic", "Propositional Logic", "RELATES_TO")
R("First-Order Logic", "Predicate Logic", "IS_A")
R("Second-Order Logic", "First-Order Logic", "RELATES_TO")
R("Boolean Algebra", "Propositional Logic", "RELATES_TO")
R("Modus Ponens", "Propositional Logic", "USED_IN")
R("Modus Tollens", "Propositional Logic", "USED_IN")
R("Resolution", "Conjunctive Normal Form", "USED_IN")
R("Horn Clause", "Prolog", "USED_IN")
R("SAT Solver", "Boolean Satisfiability", "ENABLES")
R("SMT Solver", "SAT Solver", "RELATES_TO")
R("Conjunctive Normal Form", "Boolean Satisfiability", "USED_IN")
R("Lambda Calculus", "Functional Programming", "ENABLES")
R("Curry-Howard Isomorphism", "Type Theory", "RELATES_TO")
R("Monad", "Haskell", "USED_IN")
R("Category Theory", "Abstract Algebra", "RELATES_TO")
R("Hoare Logic", "Formal Verification", "USED_IN")
R("Model Checking", "Formal Verification", "IS_A")
R("Theorem Proving", "Formal Verification", "IS_A")
R("BDD", "SAT Solver", "RELATES_TO")
R("Regular Expression", "Finite Automaton", "RELATES_TO")
R("Y Combinator", "Lambda Calculus", "PART_OF")
R("Russell's Paradox", "Set Theory", "RELATES_TO")
R("Zermelo-Fraenkel Set Theory", "Set Theory", "IS_A")
R("Gödel's Incompleteness Theorem", "Proof Theory", "RELATES_TO")
R("Rice's Theorem", "Halting Problem", "RELATES_TO")

# ============================================================
# CROSS-DOMAIN RELATIONS
# ============================================================
# CS ↔ Math
R("Hash Table", "Modular Arithmetic", "USED_IN")
R("Binary Search", "Logarithm", "RELATES_TO")
R("Merge Sort", "Recurrence Relation", "RELATES_TO")
R("Graph Theory", "Graph", "RELATES_TO")
R("Neural Network", "Linear Algebra", "USED_IN")
R("Neural Network", "Calculus", "USED_IN")
R("Machine Learning Framework", "Linear Algebra", "USED_IN")
R("Bloom Filter", "Probability Theory", "USED_IN")
R("HyperLogLog", "Probability Theory", "USED_IN")
R("PageRank", "Markov Chain", "RELATES_TO")
R("PageRank", "Eigenvalue", "RELATES_TO")
R("K-Means Clustering", "Euclidean Geometry", "RELATES_TO")
R("Support Vector Machine", "Convex Optimization", "USED_IN")
R("Transformer", "Linear Algebra", "USED_IN")
R("Convolutional Neural Network", "Fourier Transform", "RELATES_TO")
R("Elliptic Curve Cryptography", "Number Theory", "USED_IN")
R("Singular Value Decomposition", "Linear Algebra", "PART_OF")
R("Gaussian Elimination", "Linear Algebra", "PART_OF")
R("Monte Carlo Method", "Monte Carlo Algorithm", "RELATES_TO")
R("Gradient Descent", "Calculus", "USED_IN")
R("Backpropagation", "Chain Rule", "USED_IN")

# Additional cross-links
R("Redis", "LRU Cache", "USED_IN")
R("B-Tree", "PostgreSQL", "USED_IN")
R("B+ Tree", "MySQL", "USED_IN")
R("Red-Black Tree", "Java", "USED_IN")
R("Hash Table", "Python", "USED_IN")
R("Trie", "DNS", "USED_IN")
R("Segment Tree", "Competitive Programming", "USED_IN")
C("Competitive Programming", "Solving algorithmic problems under time constraints in contests")
R("Bloom Filter", "Redis", "USED_IN")
R("Skip List", "Redis", "USED_IN")
R("Dijkstra's Algorithm", "OSPF", "USED_IN")
R("Bellman-Ford Algorithm", "RIP", "USED_IN")

# OS concepts
os_concepts = {
    "Operating System": "System software managing hardware resources and providing services to programs",
    "Linux": "Open-source Unix-like operating system kernel",
    "Windows": "Microsoft's proprietary operating system family",
    "macOS": "Apple's operating system for Macintosh computers",
    "Process": "Instance of a program in execution with its own memory space",
    "Thread": "Lightweight unit of execution within a process sharing its memory",
    "Mutex": "Mutual exclusion lock ensuring only one thread accesses a resource",
    "Semaphore": "Signaling mechanism controlling access to shared resources by multiple threads",
    "Deadlock": "State where processes block each other waiting for resources held by others",
    "Context Switch": "Saving and restoring state when switching between processes or threads",
    "Scheduler": "OS component deciding which process runs next on the CPU",
    "Virtual Memory": "Memory management mapping virtual addresses to physical using paging",
    "Page Fault": "Interrupt when a process accesses a page not in physical memory",
    "File System": "Method for organizing and storing files on storage devices",
    "ext4": "Fourth extended file system — default Linux filesystem",
    "NTFS": "New Technology File System — Windows default filesystem",
    "APFS": "Apple File System — modern filesystem for macOS and iOS",
    "ZFS": "Advanced filesystem with built-in volume management and data integrity",
    "Btrfs": "B-tree filesystem for Linux with snapshots and RAID support",
    "Kernel": "Core of OS managing CPU, memory, and device communication",
    "System Call": "Interface between user programs and the operating system kernel",
    "Device Driver": "Software enabling OS to communicate with hardware devices",
    "Bootloader": "Program that loads the operating system into memory at startup",
    "GRUB": "Grand Unified Bootloader — common Linux bootloader",
    "Containerization": "Lightweight virtualization isolating applications in containers",
    "Virtualization": "Creating virtual versions of hardware platforms and operating systems",
    "Hypervisor": "Software creating and managing virtual machines",
    "KVM": "Kernel-based Virtual Machine — Linux virtualization infrastructure",
    "VMware": "Enterprise virtualization platform for running multiple OS instances",
    "Proxmox": "Open-source server virtualization management platform",
    "systemd": "Linux init system and service manager",
    "cgroup": "Linux kernel feature limiting and isolating resource usage of process groups",
    "namespace": "Linux kernel feature providing process isolation for containerization",
}

for name, defn in os_concepts.items():
    C(name, defn)

R("Linux", "Operating System", "IS_A")
R("Windows", "Operating System", "IS_A")
R("macOS", "Operating System", "IS_A")
R("Thread", "Process", "PART_OF")
R("Mutex", "Thread", "USED_IN")
R("Semaphore", "Thread", "USED_IN")
R("Deadlock", "Mutex", "RELATES_TO")
R("Virtual Memory", "Page Table", "USED_IN")
R("ext4", "File System", "IS_A")
R("NTFS", "File System", "IS_A")
R("ZFS", "File System", "IS_A")
R("Kernel", "Operating System", "PART_OF")
R("Docker", "Containerization", "IS_A")
R("Docker", "cgroup", "USED_IN")
R("Docker", "namespace", "USED_IN")
R("Kubernetes", "Containerization", "USED_IN")
R("KVM", "Hypervisor", "IS_A")
R("VMware", "Hypervisor", "IS_A")
R("Proxmox", "KVM", "USED_IN")

# Design Patterns
patterns = {
    "Design Pattern": "Reusable solution to a commonly occurring problem in software design",
    "Singleton Pattern": "Ensuring a class has only one instance with global access",
    "Factory Pattern": "Creating objects without specifying exact class to instantiate",
    "Abstract Factory": "Interface for creating families of related objects",
    "Builder Pattern": "Constructing complex objects step by step",
    "Prototype Pattern": "Creating objects by cloning an existing instance",
    "Adapter Pattern": "Converting interface of a class into another interface clients expect",
    "Bridge Pattern": "Separating abstraction from implementation so both can vary",
    "Composite Pattern": "Composing objects into tree structures for part-whole hierarchies",
    "Decorator Pattern": "Dynamically adding behavior to objects without modifying their class",
    "Facade Pattern": "Providing simplified interface to a complex subsystem",
    "Flyweight Pattern": "Sharing common state between multiple objects to save memory",
    "Proxy Pattern": "Providing surrogate or placeholder for another object",
    "Observer Pattern": "Defining one-to-many dependency so dependents update automatically",
    "Strategy Pattern": "Defining family of algorithms and making them interchangeable",
    "Command Pattern": "Encapsulating a request as an object for parameterization",
    "Iterator Pattern": "Providing way to access elements sequentially without exposing internals",
    "Mediator Pattern": "Defining object encapsulating how a set of objects interact",
    "Memento Pattern": "Capturing and restoring object's internal state",
    "State Pattern": "Allowing object to alter behavior when internal state changes",
    "Template Method": "Defining algorithm skeleton deferring some steps to subclasses",
    "Visitor Pattern": "Separating algorithm from object structure it operates on",
    "Chain of Responsibility": "Passing request along chain of handlers until one handles it",
    "MVC Pattern": "Model-View-Controller — separating data, presentation, and logic",
    "MVVM Pattern": "Model-View-ViewModel — variation of MVC for data binding",
    "Repository Pattern": "Abstracting data access behind a collection-like interface",
    "Dependency Injection": "Supplying dependencies to objects rather than having them create dependencies",
    "Event-Driven Architecture": "Architecture where flow is determined by events",
    "Microservices Architecture": "Architecture decomposing application into small independent services",
    "Monolithic Architecture": "Architecture where application is a single deployable unit",
    "Serverless Architecture": "Architecture where cloud provider manages server infrastructure",
    "CQRS": "Command Query Responsibility Segregation — separating read and write models",
    "Event Sourcing": "Storing state changes as sequence of events",
    "Domain-Driven Design": "Software design approach focusing on core domain and domain logic",
    "Clean Architecture": "Architecture separating concerns with dependency rule pointing inward",
    "Hexagonal Architecture": "Architecture isolating core logic from external concerns via ports and adapters",
}

for name, defn in patterns.items():
    C(name, defn)
    if name != "Design Pattern":
        R(name, "Design Pattern", "IS_A")
R("Design Pattern", "Object-Oriented Programming", "USED_IN")
R("Microservices Architecture", "Docker", "USED_IN")
R("Microservices Architecture", "Kubernetes", "USED_IN")
R("Dependency Injection", "Spring Framework", "USED_IN")
R("MVC Pattern", "Ruby on Rails", "USED_IN")
R("MVC Pattern", "Django", "USED_IN")
R("MVVM Pattern", "Angular", "USED_IN")

# Software Engineering concepts
se = {
    "Agile": "Iterative software development methodology emphasizing collaboration and adaptability",
    "Scrum": "Agile framework with sprints, daily standups, and defined roles",
    "Kanban": "Visual workflow management method limiting work in progress",
    "DevOps": "Practices combining software development and IT operations",
    "CI/CD": "Continuous Integration and Continuous Deployment — automating build and release",
    "Test-Driven Development": "Development practice writing tests before implementation",
    "Behavior-Driven Development": "Extension of TDD using natural language specifications",
    "Code Review": "Systematic examination of source code by peers",
    "Refactoring": "Restructuring existing code without changing its external behavior",
    "Technical Debt": "Implied cost of future rework from choosing quick solutions now",
    "API": "Application Programming Interface — contract for software component interaction",
    "REST API": "API following REST architectural constraints using HTTP methods",
    "GraphQL API": "API using GraphQL for flexible data querying",
    "Webhook": "HTTP callback triggered by an event to notify external systems",
    "Idempotency": "Property where repeating an operation produces the same result",
    "ACID Properties": "Atomicity, Consistency, Isolation, Durability — database transaction guarantees",
    "CAP Theorem": "Distributed system can guarantee at most 2 of: Consistency, Availability, Partition tolerance",
    "BASE Properties": "Basically Available, Soft state, Eventually consistent — alternative to ACID",
    "Eventual Consistency": "Guarantee that all replicas converge to the same value given no new updates",
    "Sharding": "Horizontal partitioning of data across multiple database instances",
    "Replication": "Copying data across multiple nodes for availability and fault tolerance",
    "Load Balancing": "Distributing workload across multiple computing resources",
    "Caching": "Storing frequently accessed data in fast storage for quick retrieval",
    "CDN": "Content Delivery Network — geographically distributed cache for web content",
    "Rate Limiting": "Controlling the rate of requests to prevent abuse",
    "Circuit Breaker": "Pattern preventing cascading failures by failing fast",
    "Retry Pattern": "Automatically retrying failed operations with backoff",
    "Saga Pattern": "Managing distributed transactions through a sequence of local transactions",
    "Two-Phase Commit": "Protocol ensuring all nodes agree to commit or abort a distributed transaction",
    "Consensus Algorithm": "Algorithm achieving agreement among distributed nodes",
    "Raft": "Consensus algorithm for managing replicated log in distributed systems",
    "Paxos": "Consensus algorithm for achieving agreement in unreliable networks",
    "Blockchain": "Distributed immutable ledger using cryptographic hashing and consensus",
    "Ethereum": "Decentralized platform for smart contracts and dApps",
    "Bitcoin": "First decentralized cryptocurrency using proof-of-work consensus",
    "Smart Contract": "Self-executing contract with terms written in code on blockchain",
    "Proof of Work": "Consensus mechanism requiring computational effort to validate",
    "Proof of Stake": "Consensus mechanism based on validator's stake in the network",
    "Google": "Technology company and search engine pioneer",
}

for name, defn in se.items():
    C(name, defn)

R("Scrum", "Agile", "IS_A"); R("Kanban", "Agile", "IS_A")
R("DevOps", "CI/CD", "ENABLES")
R("CAP Theorem", "Distributed Algorithm", "RELATES_TO")
R("Raft", "Consensus Algorithm", "IS_A")
R("Paxos", "Consensus Algorithm", "IS_A")
R("Blockchain", "SHA-256", "USED_IN")
R("Bitcoin", "Blockchain", "USED_IN")
R("Ethereum", "Blockchain", "USED_IN")
R("Smart Contract", "Ethereum", "USED_IN")
R("Smart Contract", "Solidity", "USED_IN")
R("CDN", "Cloudflare", "RELATES_TO")
R("REST API", "HTTP", "USED_IN")
R("ACID Properties", "Relational Database", "HAS_PROPERTY")
R("Sharding", "MongoDB", "USED_IN")
R("Replication", "PostgreSQL", "USED_IN")

# Write output
data = {"concepts": concepts, "relations": relations}
print(f"Concepts: {len(concepts)}")
print(f"Relations: {len(relations)}")

with open("/home/hirschpekf/brain19/data/wave7_cs_math.json", "w") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
print("Written to wave7_cs_math.json")