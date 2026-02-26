//! Noise filter for built-in/standard library function calls
//!
//! Filters out common built-in function names across all supported languages
//! to prevent polluting the call graph with noise (e.g., console.log, print, malloc).

use std::collections::HashSet;
use std::sync::LazyLock;

/// Set of built-in/standard library function names to filter from the call graph.
/// Organized by language for maintainability.
static BUILT_IN_NAMES: LazyLock<HashSet<&'static str>> = LazyLock::new(|| {
    let mut s = HashSet::with_capacity(400);

    // =========================================================================
    // JavaScript / TypeScript
    // =========================================================================
    // Console
    s.insert("log");
    s.insert("warn");
    s.insert("error");
    s.insert("info");
    s.insert("debug");
    s.insert("trace");
    s.insert("assert");
    s.insert("dir");
    s.insert("table");
    s.insert("time");
    s.insert("timeEnd");
    s.insert("timeLog");
    s.insert("clear");
    s.insert("count");
    s.insert("countReset");
    s.insert("group");
    s.insert("groupEnd");
    s.insert("groupCollapsed");
    // Global functions
    s.insert("parseInt");
    s.insert("parseFloat");
    s.insert("isNaN");
    s.insert("isFinite");
    s.insert("encodeURI");
    s.insert("encodeURIComponent");
    s.insert("decodeURI");
    s.insert("decodeURIComponent");
    s.insert("setTimeout");
    s.insert("setInterval");
    s.insert("clearTimeout");
    s.insert("clearInterval");
    s.insert("requestAnimationFrame");
    s.insert("cancelAnimationFrame");
    s.insert("fetch");
    s.insert("alert");
    s.insert("confirm");
    s.insert("prompt");
    s.insert("atob");
    s.insert("btoa");
    // Promise
    s.insert("resolve");
    s.insert("reject");
    s.insert("then");
    s.insert("catch");
    s.insert("finally");
    s.insert("all");
    s.insert("allSettled");
    s.insert("race");
    s.insert("any");
    // Array / Object / String methods
    s.insert("push");
    s.insert("pop");
    s.insert("shift");
    s.insert("unshift");
    s.insert("splice");
    s.insert("slice");
    s.insert("concat");
    s.insert("join");
    s.insert("indexOf");
    s.insert("lastIndexOf");
    s.insert("includes");
    s.insert("find");
    s.insert("findIndex");
    s.insert("filter");
    s.insert("map");
    s.insert("reduce");
    s.insert("reduceRight");
    s.insert("forEach");
    s.insert("every");
    s.insert("some");
    s.insert("sort");
    s.insert("reverse");
    s.insert("flat");
    s.insert("flatMap");
    s.insert("fill");
    s.insert("copyWithin");
    s.insert("entries");
    s.insert("keys");
    s.insert("values");
    s.insert("from");
    s.insert("of");
    s.insert("isArray");
    s.insert("assign");
    s.insert("freeze");
    s.insert("defineProperty");
    s.insert("getOwnPropertyNames");
    s.insert("hasOwnProperty");
    s.insert("toString");
    s.insert("valueOf");
    s.insert("toFixed");
    s.insert("toPrecision");
    s.insert("toLocaleString");
    s.insert("charAt");
    s.insert("charCodeAt");
    s.insert("codePointAt");
    s.insert("startsWith");
    s.insert("endsWith");
    s.insert("trim");
    s.insert("trimStart");
    s.insert("trimEnd");
    s.insert("padStart");
    s.insert("padEnd");
    s.insert("repeat");
    s.insert("replace");
    s.insert("replaceAll");
    s.insert("split");
    s.insert("substring");
    s.insert("toLowerCase");
    s.insert("toUpperCase");
    s.insert("match");
    s.insert("matchAll");
    s.insert("search");
    s.insert("normalize");
    s.insert("localeCompare");
    // JSON
    s.insert("parse");
    s.insert("stringify");
    // Math
    s.insert("abs");
    s.insert("ceil");
    s.insert("floor");
    s.insert("round");
    s.insert("max");
    s.insert("min");
    s.insert("pow");
    s.insert("sqrt");
    s.insert("random");
    s.insert("trunc");
    s.insert("sign");
    // Map / Set (only unambiguous ones, not get/set which are common method names)
    s.insert("has");
    s.insert("delete");
    s.insert("size");
    // React hooks
    s.insert("useState");
    s.insert("useEffect");
    s.insert("useContext");
    s.insert("useReducer");
    s.insert("useCallback");
    s.insert("useMemo");
    s.insert("useRef");
    s.insert("useLayoutEffect");
    s.insert("useImperativeHandle");
    s.insert("useDebugValue");
    s.insert("useTransition");
    s.insert("useDeferredValue");
    s.insert("useId");
    // DOM
    s.insert("getElementById");
    s.insert("querySelector");
    s.insert("querySelectorAll");
    s.insert("createElement");
    s.insert("appendChild");
    s.insert("removeChild");
    s.insert("addEventListener");
    s.insert("removeEventListener");
    s.insert("preventDefault");
    s.insert("stopPropagation");
    s.insert("setAttribute");
    s.insert("getAttribute");
    s.insert("classList");
    // Node.js
    s.insert("require");
    s.insert("emit");
    s.insert("on");
    s.insert("once");
    s.insert("pipe");
    s.insert("write");
    s.insert("end");
    s.insert("listen");
    s.insert("close");
    s.insert("nextTick");

    // =========================================================================
    // Python
    // =========================================================================
    s.insert("print");
    s.insert("len");
    s.insert("range");
    s.insert("enumerate");
    s.insert("zip");
    s.insert("type");
    s.insert("isinstance");
    s.insert("issubclass");
    s.insert("id");
    s.insert("hash");
    s.insert("repr");
    s.insert("str");
    s.insert("int");
    s.insert("float");
    s.insert("bool");
    s.insert("list");
    s.insert("dict");
    s.insert("tuple");
    s.insert("bytes");
    s.insert("bytearray");
    s.insert("memoryview");
    s.insert("frozenset");
    s.insert("complex");
    s.insert("input");
    s.insert("open");
    s.insert("iter");
    s.insert("next");
    s.insert("sorted");
    s.insert("reversed");
    s.insert("getattr");
    s.insert("setattr");
    s.insert("delattr");
    s.insert("hasattr");
    s.insert("callable");
    s.insert("super");
    s.insert("property");
    s.insert("staticmethod");
    s.insert("classmethod");
    s.insert("vars");
    s.insert("dir");
    s.insert("globals");
    s.insert("locals");
    s.insert("any");
    s.insert("all");
    s.insert("sum");
    s.insert("min");
    s.insert("max");
    s.insert("abs");
    s.insert("round");
    s.insert("divmod");
    s.insert("pow");
    s.insert("ord");
    s.insert("chr");
    s.insert("hex");
    s.insert("oct");
    s.insert("bin");
    s.insert("format");
    s.insert("exec");
    s.insert("eval");
    s.insert("compile");
    s.insert("breakpoint");
    s.insert("__import__");
    // Python string/list methods
    s.insert("append");
    s.insert("extend");
    s.insert("insert");
    s.insert("remove");
    s.insert("pop");
    s.insert("clear");
    s.insert("copy");
    s.insert("update");
    s.insert("items");
    s.insert("strip");
    s.insert("lstrip");
    s.insert("rstrip");
    s.insert("upper");
    s.insert("lower");
    s.insert("title");
    s.insert("capitalize");
    s.insert("encode");
    s.insert("decode");
    s.insert("format_map");
    s.insert("isdigit");
    s.insert("isalpha");
    s.insert("isalnum");

    // =========================================================================
    // C / C++
    // =========================================================================
    // stdio
    s.insert("printf");
    s.insert("fprintf");
    s.insert("sprintf");
    s.insert("snprintf");
    s.insert("scanf");
    s.insert("fscanf");
    s.insert("sscanf");
    s.insert("puts");
    s.insert("fputs");
    s.insert("fgets");
    s.insert("getchar");
    s.insert("putchar");
    s.insert("fopen");
    s.insert("fclose");
    s.insert("fread");
    s.insert("fwrite");
    s.insert("fseek");
    s.insert("ftell");
    s.insert("fflush");
    s.insert("rewind");
    s.insert("perror");
    // stdlib
    s.insert("malloc");
    s.insert("calloc");
    s.insert("realloc");
    s.insert("free");
    s.insert("exit");
    s.insert("abort");
    s.insert("atexit");
    s.insert("atoi");
    s.insert("atof");
    s.insert("atol");
    s.insert("strtol");
    s.insert("strtod");
    s.insert("strtoul");
    s.insert("qsort");
    s.insert("bsearch");
    s.insert("rand");
    s.insert("srand");
    s.insert("system");
    s.insert("getenv");
    // string.h
    s.insert("strlen");
    s.insert("strcpy");
    s.insert("strncpy");
    s.insert("strcat");
    s.insert("strncat");
    s.insert("strcmp");
    s.insert("strncmp");
    s.insert("strchr");
    s.insert("strrchr");
    s.insert("strstr");
    s.insert("strtok");
    s.insert("memcpy");
    s.insert("memmove");
    s.insert("memset");
    s.insert("memcmp");
    // C++ specific
    s.insert("cout");
    s.insert("cerr");
    s.insert("endl");
    s.insert("make_shared");
    s.insert("make_unique");
    s.insert("make_pair");
    s.insert("make_tuple");
    s.insert("move");
    s.insert("forward");
    s.insert("swap");
    s.insert("begin");
    s.insert("end");
    s.insert("size");
    s.insert("empty");
    s.insert("push_back");
    s.insert("pop_back");
    s.insert("emplace_back");
    s.insert("emplace");
    s.insert("erase");
    s.insert("front");
    s.insert("back");
    s.insert("at");
    s.insert("reserve");
    s.insert("resize");
    s.insert("capacity");
    s.insert("shrink_to_fit");
    s.insert("data");
    s.insert("c_str");
    s.insert("substr");
    s.insert("npos");
    s.insert("dynamic_cast");
    s.insert("static_cast");
    s.insert("reinterpret_cast");
    s.insert("const_cast");
    s.insert("throw");
    s.insert("new");
    s.insert("sizeof");
    s.insert("alignof");
    s.insert("decltype");
    s.insert("typeid");
    // assert
    s.insert("assert");
    s.insert("static_assert");
    // POSIX
    s.insert("read");
    s.insert("close");
    s.insert("fork");
    s.insert("wait");
    s.insert("waitpid");
    s.insert("kill");
    s.insert("signal");
    s.insert("sleep");
    s.insert("usleep");

    // =========================================================================
    // Java
    // =========================================================================
    s.insert("println");
    s.insert("equals");
    s.insert("hashCode");
    s.insert("compareTo");
    s.insert("getClass");
    s.insert("notify");
    s.insert("notifyAll");
    s.insert("clone");
    s.insert("finalize");
    s.insert("length");
    s.insert("charAt");
    s.insert("toCharArray");
    s.insert("getBytes");
    s.insert("isEmpty");
    s.insert("contains");
    s.insert("containsKey");
    s.insert("containsValue");
    s.insert("put");
    s.insert("putAll");
    s.insert("entrySet");
    s.insert("keySet");
    s.insert("toArray");
    s.insert("iterator");
    s.insert("hasNext");
    s.insert("getName");
    s.insert("getMessage");
    s.insert("printStackTrace");
    s.insert("currentThread");
    s.insert("start");
    s.insert("run");
    s.insert("interrupt");
    s.insert("isAlive");
    s.insert("currentTimeMillis");
    s.insert("nanoTime");
    s.insert("arraycopy");
    s.insert("gc");

    // =========================================================================
    // Go
    // =========================================================================
    s.insert("make");
    s.insert("new");
    s.insert("append");
    s.insert("copy");
    s.insert("cap");
    s.insert("len");
    s.insert("close");
    s.insert("delete");
    s.insert("panic");
    s.insert("recover");
    s.insert("real");
    s.insert("imag");
    s.insert("Println");
    s.insert("Printf");
    s.insert("Sprintf");
    s.insert("Fprintf");
    s.insert("Errorf");
    s.insert("Print");
    s.insert("Fatalf");
    s.insert("Fatal");
    s.insert("Panicf");
    s.insert("Error");
    s.insert("New");
    s.insert("Wrap");
    s.insert("Unwrap");
    s.insert("Is");
    s.insert("As");
    s.insert("String");
    s.insert("Bytes");
    s.insert("Len");
    s.insert("Cap");

    // =========================================================================
    // Rust
    // =========================================================================
    // Macros (appear without !)
    s.insert("println");
    s.insert("eprintln");
    s.insert("dbg");
    s.insert("todo");
    s.insert("unimplemented");
    s.insert("unreachable");
    s.insert("panic");
    s.insert("assert_eq");
    s.insert("assert_ne");
    s.insert("debug_assert");
    s.insert("debug_assert_eq");
    s.insert("debug_assert_ne");
    s.insert("write");
    s.insert("writeln");
    s.insert("vec");
    s.insert("format");
    s.insert("cfg");
    s.insert("env");
    s.insert("include_str");
    s.insert("include_bytes");
    s.insert("file");
    s.insert("line");
    s.insert("column");
    s.insert("module_path");
    s.insert("stringify");
    s.insert("concat");
    s.insert("matches");
    // Std methods
    s.insert("unwrap");
    s.insert("expect");
    s.insert("unwrap_or");
    s.insert("unwrap_or_else");
    s.insert("unwrap_or_default");
    s.insert("ok");
    s.insert("err");
    s.insert("is_some");
    s.insert("is_none");
    s.insert("is_ok");
    s.insert("is_err");
    s.insert("map");
    s.insert("map_err");
    s.insert("and_then");
    s.insert("or_else");
    s.insert("filter");
    s.insert("flatten");
    s.insert("collect");
    s.insert("iter");
    s.insert("into_iter");
    s.insert("iter_mut");
    s.insert("next");
    s.insert("chain");
    s.insert("zip");
    s.insert("enumerate");
    s.insert("take");
    s.insert("skip");
    s.insert("peekable");
    s.insert("cloned");
    s.insert("copied");
    s.insert("clone");
    s.insert("to_string");
    s.insert("to_owned");
    s.insert("as_ref");
    s.insert("as_mut");
    s.insert("as_str");
    s.insert("as_bytes");
    s.insert("as_slice");
    s.insert("into");
    s.insert("from");
    s.insert("default");
    s.insert("new");
    s.insert("with_capacity");
    s.insert("push");
    s.insert("pop");
    s.insert("insert");
    s.insert("remove");
    s.insert("contains");
    s.insert("is_empty");
    s.insert("len");
    s.insert("clear");
    s.insert("extend");
    s.insert("drain");
    s.insert("retain");
    s.insert("sort");
    s.insert("sort_by");
    s.insert("sort_by_key");
    s.insert("dedup");
    s.insert("truncate");
    s.insert("split_at");
    s.insert("windows");
    s.insert("chunks");
    s.insert("entry");
    s.insert("or_insert");
    s.insert("or_insert_with");
    s.insert("lock");
    s.insert("read");
    s.insert("write");
    s.insert("try_lock");
    s.insert("send");
    s.insert("recv");
    s.insert("try_recv");
    s.insert("spawn");
    s.insert("join");
    s.insert("await");

    // =========================================================================
    // PHP
    // =========================================================================
    s.insert("echo");
    s.insert("var_dump");
    s.insert("print_r");
    s.insert("var_export");
    s.insert("die");
    s.insert("isset");
    s.insert("unset");
    s.insert("empty");
    s.insert("array_push");
    s.insert("array_pop");
    s.insert("array_shift");
    s.insert("array_unshift");
    s.insert("array_merge");
    s.insert("array_keys");
    s.insert("array_values");
    s.insert("array_map");
    s.insert("array_filter");
    s.insert("array_reduce");
    s.insert("array_slice");
    s.insert("array_splice");
    s.insert("array_search");
    s.insert("in_array");
    s.insert("count");
    s.insert("array_key_exists");
    s.insert("array_unique");
    s.insert("array_flip");
    s.insert("array_reverse");
    s.insert("array_combine");
    s.insert("implode");
    s.insert("explode");
    s.insert("str_replace");
    s.insert("str_contains");
    s.insert("str_starts_with");
    s.insert("str_ends_with");
    s.insert("substr");
    s.insert("strtolower");
    s.insert("strtoupper");
    s.insert("ucfirst");
    s.insert("lcfirst");
    s.insert("ucwords");
    s.insert("nl2br");
    s.insert("htmlspecialchars");
    s.insert("htmlentities");
    s.insert("strip_tags");
    s.insert("number_format");
    s.insert("sprintf");
    s.insert("preg_match");
    s.insert("preg_replace");
    s.insert("preg_split");
    s.insert("json_encode");
    s.insert("json_decode");
    s.insert("file_get_contents");
    s.insert("file_put_contents");
    s.insert("file_exists");
    s.insert("is_file");
    s.insert("is_dir");
    s.insert("mkdir");
    s.insert("rmdir");
    s.insert("unlink");
    s.insert("rename");
    s.insert("glob");
    s.insert("realpath");
    s.insert("basename");
    s.insert("dirname");
    s.insert("pathinfo");
    s.insert("class_exists");
    s.insert("method_exists");
    s.insert("property_exists");
    s.insert("function_exists");
    s.insert("get_class");
    s.insert("is_a");

    // =========================================================================
    // Ruby
    // =========================================================================
    s.insert("puts");
    s.insert("p");
    s.insert("pp");
    s.insert("print");
    s.insert("raise");
    s.insert("require");
    s.insert("require_relative");
    s.insert("include");
    s.insert("extend");
    s.insert("prepend");
    s.insert("attr_reader");
    s.insert("attr_writer");
    s.insert("attr_accessor");
    s.insert("each");
    s.insert("each_with_index");
    s.insert("each_with_object");
    s.insert("map");
    s.insert("select");
    s.insert("reject");
    s.insert("detect");
    s.insert("collect");
    s.insert("inject");
    s.insert("reduce");
    s.insert("flat_map");
    s.insert("compact");
    s.insert("uniq");
    s.insert("first");
    s.insert("last");
    s.insert("count");
    s.insert("any?");
    s.insert("all?");
    s.insert("none?");
    s.insert("empty?");
    s.insert("nil?");
    s.insert("frozen?");
    s.insert("respond_to?");
    s.insert("is_a?");
    s.insert("kind_of?");
    s.insert("instance_of?");
    s.insert("freeze");
    s.insert("dup");
    s.insert("tap");
    s.insert("then");
    s.insert("yield_self");
    s.insert("send");
    s.insert("public_send");
    s.insert("method_missing");
    s.insert("define_method");
    s.insert("class_eval");
    s.insert("instance_eval");
    s.insert("new");
    s.insert("initialize");
    s.insert("to_s");
    s.insert("to_i");
    s.insert("to_f");
    s.insert("to_a");
    s.insert("to_h");
    s.insert("to_sym");
    s.insert("inspect");

    // =========================================================================
    // Kotlin
    // =========================================================================
    s.insert("println");
    s.insert("print");
    s.insert("require");
    s.insert("requireNotNull");
    s.insert("check");
    s.insert("checkNotNull");
    s.insert("error");
    s.insert("TODO");
    s.insert("let");
    s.insert("run");
    s.insert("with");
    s.insert("apply");
    s.insert("also");
    s.insert("takeIf");
    s.insert("takeUnless");
    s.insert("repeat");
    s.insert("lazy");
    s.insert("listOf");
    s.insert("mutableListOf");
    s.insert("mapOf");
    s.insert("mutableMapOf");
    s.insert("setOf");
    s.insert("mutableSetOf");
    s.insert("arrayOf");
    s.insert("emptyList");
    s.insert("emptyMap");
    s.insert("emptySet");
    s.insert("sequenceOf");
    s.insert("buildList");
    s.insert("buildMap");
    s.insert("buildSet");
    s.insert("hashMapOf");
    s.insert("linkedMapOf");
    s.insert("sortedMapOf");

    // =========================================================================
    // Swift
    // =========================================================================
    s.insert("print");
    s.insert("debugPrint");
    s.insert("dump");
    s.insert("fatalError");
    s.insert("precondition");
    s.insert("preconditionFailure");
    s.insert("assertionFailure");
    s.insert("type");
    s.insert("min");
    s.insert("max");
    s.insert("abs");
    s.insert("stride");
    s.insert("zip");
    s.insert("sequence");
    s.insert("repeatElement");
    s.insert("withUnsafePointer");
    s.insert("withUnsafeMutablePointer");
    s.insert("withUnsafeBytes");
    s.insert("DispatchQueue");
    s.insert("NotificationCenter");
    // Swift collection methods
    s.insert("append");
    s.insert("insert");
    s.insert("remove");
    s.insert("removeAll");
    s.insert("contains");
    s.insert("firstIndex");
    s.insert("lastIndex");
    s.insert("compactMap");
    s.insert("flatMap");
    s.insert("sorted");
    s.insert("enumerated");

    // =========================================================================
    // Bash
    // =========================================================================
    s.insert("echo");
    s.insert("printf");
    s.insert("read");
    s.insert("test");
    s.insert("exit");
    s.insert("return");
    s.insert("export");
    s.insert("source");
    s.insert("eval");
    s.insert("exec");
    s.insert("set");
    s.insert("unset");
    s.insert("shift");
    s.insert("trap");
    s.insert("wait");
    s.insert("cd");
    s.insert("pwd");
    s.insert("ls");
    s.insert("cp");
    s.insert("mv");
    s.insert("rm");
    s.insert("mkdir");
    s.insert("rmdir");
    s.insert("cat");
    s.insert("grep");
    s.insert("sed");
    s.insert("awk");
    s.insert("find");
    s.insert("xargs");
    s.insert("sort");
    s.insert("uniq");
    s.insert("wc");
    s.insert("head");
    s.insert("tail");
    s.insert("cut");
    s.insert("tr");
    s.insert("tee");
    s.insert("true");
    s.insert("false");
    s.insert("local");
    s.insert("declare");
    s.insert("typeset");
    s.insert("readonly");

    // =========================================================================
    // Testing frameworks (cross-language)
    // =========================================================================
    s.insert("describe");
    s.insert("it");
    s.insert("test");
    s.insert("expect");
    s.insert("beforeEach");
    s.insert("afterEach");
    s.insert("beforeAll");
    s.insert("afterAll");
    s.insert("mock");
    s.insert("jest");
    s.insert("spy");
    s.insert("stub");
    s.insert("verify");
    s.insert("when");
    s.insert("given");
    s.insert("should");
    s.insert("toBe");
    s.insert("toEqual");
    s.insert("toContain");
    s.insert("toThrow");
    s.insert("toHaveBeenCalled");
    s.insert("toHaveBeenCalledWith");
    s.insert("toMatchSnapshot");
    s.insert("assertEquals");
    s.insert("assertTrue");
    s.insert("assertFalse");
    s.insert("assertNull");
    s.insert("assertNotNull");
    s.insert("assertThrows");
    s.insert("fail");

    s
});

/// Get a reference to the full set of built-in names (for Neo4j cleanup queries).
pub fn builtin_names() -> &'static HashSet<&'static str> {
    &BUILT_IN_NAMES
}

/// Check if a function call name is a built-in/noise that should be filtered.
///
/// Returns `true` if the name should be excluded from the call graph.
pub fn is_builtin_call(name: &str) -> bool {
    // Strip method receiver prefix for qualified names (e.g., "fmt::Println" → "Println")
    let base_name = name.rsplit("::").next().unwrap_or(name);
    let base_name = base_name.rsplit('.').next().unwrap_or(base_name);

    BUILT_IN_NAMES.contains(base_name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builtin_count() {
        // Verify we have a substantial number of entries
        assert!(
            BUILT_IN_NAMES.len() >= 300,
            "Expected 300+ built-in names, got {}",
            BUILT_IN_NAMES.len()
        );
    }

    #[test]
    fn test_common_builtins_detected() {
        // JS/TS
        assert!(is_builtin_call("log"));
        assert!(is_builtin_call("console.log"));
        assert!(is_builtin_call("forEach"));
        assert!(is_builtin_call("useState"));
        assert!(is_builtin_call("parseInt"));

        // Python
        assert!(is_builtin_call("print"));
        assert!(is_builtin_call("len"));
        assert!(is_builtin_call("range"));
        assert!(is_builtin_call("isinstance"));

        // C/C++
        assert!(is_builtin_call("printf"));
        assert!(is_builtin_call("malloc"));
        assert!(is_builtin_call("free"));
        assert!(is_builtin_call("strlen"));

        // Java
        assert!(is_builtin_call("println"));
        assert!(is_builtin_call("equals"));
        assert!(is_builtin_call("toString"));

        // Go
        assert!(is_builtin_call("make"));
        assert!(is_builtin_call("Println"));
        assert!(is_builtin_call("fmt::Println"));
        assert!(is_builtin_call("Errorf"));

        // Rust
        assert!(is_builtin_call("unwrap"));
        assert!(is_builtin_call("expect"));
        assert!(is_builtin_call("collect"));
        assert!(is_builtin_call("clone"));

        // PHP
        assert!(is_builtin_call("var_dump"));
        assert!(is_builtin_call("array_push"));
        assert!(is_builtin_call("json_encode"));
    }

    #[test]
    fn test_real_function_names_not_filtered() {
        assert!(!is_builtin_call("processData"));
        assert!(!is_builtin_call("handleRequest"));
        assert!(!is_builtin_call("calculateTotal"));
        assert!(!is_builtin_call("validateInput"));
        assert!(!is_builtin_call("fetchUserData"));
        assert!(!is_builtin_call("renderComponent"));
        assert!(!is_builtin_call("parseConfig"));
        assert!(!is_builtin_call("serializeResponse"));
        assert!(!is_builtin_call("authenticateUser"));
        assert!(!is_builtin_call("dispatchEvent"));
        assert!(!is_builtin_call("buildQuery"));
        assert!(!is_builtin_call("transformPayload"));
    }

    #[test]
    fn test_qualified_names_stripped() {
        assert!(is_builtin_call("fmt::Println"));
        assert!(is_builtin_call("console.log"));
        assert!(is_builtin_call("System.out.println"));
        assert!(is_builtin_call("std::make_shared"));
    }
}
