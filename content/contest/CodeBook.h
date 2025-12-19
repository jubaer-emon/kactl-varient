/**
 * Author: Mashrur Mahmood
 * Date: 2025-12-24
 * License: CC0
 * Source: vairous
 * Description: na
 * Status: n/a
 */
#pragma once

// DSU
vector<ll> parent;
vector<ll> sizes;
void make_set(ll v) {
    parent[v] = v;
    sizes[v] = 1;
}

ll find_set(ll v) {
    if (v == parent[v])
        return v;
    return parent[v] = find_set(parent[v]);
}

void union_sets(ll a, ll b) {
    a = find_set(a);
    b = find_set(b);
    if (a != b){
        if (sizes[a] < sizes[b])
            swap(a, b);
        parent[b] = a;
        sizes[a] += sizes[b];
    }
}

// Linked List with Union
struct node{
    node *nxt = nullptr;
    ll val;
    node(ll _v){
        val = _v;
    }
};
struct LI{
    node *head = nullptr, *tail = nullptr;
};

vector<LI*> p;
// b gonna own
void unionp(ll a, ll b){
    if (p[b]->head == nullptr)
        p[b]->head = p[a]->head;
    else
        p[b]->tail->nxt = p[a]->head;

    p[b]->tail = p[a]->tail;
    p[a]->head = nullptr;
    p[a]->tail = nullptr;
}

/*
for (ll i = 0; i < n; i++){
    node *nd = new node(i);
    LI *l = new LI;
    l->head = nd;
    l->tail = nd;
    p.push_back(l);
}
*/

// INVERSION COUNT
ll merge(vector<ll> &arr, vector<ll> &temp, ll left, ll mid, ll right) {
    ll i = left, j = mid, k = left;
    ll inv_count = 0;

    while (i <= mid - 1 && j <= right) {
        if (arr[i] <= arr[j]) {
            temp[k++] = arr[i++];
        } else {
            temp[k++] = arr[j++];
            inv_count += (mid - i);  // all remaining in left half are > arr[j]
        }
    }

    while (i <= mid - 1)
        temp[k++] = arr[i++];
    while (j <= right)
        temp[k++] = arr[j++];

    for (ll idx = left; idx <= right; idx++)
        arr[idx] = temp[idx];

    return inv_count;
}

ll _mergeSort(vector<ll> &arr, vector<ll> &temp, ll left, ll right) {
    ll inv_count = 0;
    if (right > left) {
        ll mid = (left + right) / 2;

        inv_count += _mergeSort(arr, temp, left, mid);
        inv_count += _mergeSort(arr, temp, mid + 1, right);
        inv_count += merge(arr, temp, left, mid + 1, right);
    }
    return inv_count;
}

ll mergeSort(vector<ll> &arr) {
    vector<ll> temp(arr.size());
    return _mergeSort(arr, temp, 0, arr.size() - 1);
}

// MOS ALGO
void remove_el(idx){

}
void add_el(idx){

}
ll get_answer(){

};

ll block_size;

struct Query {
    ll l, r, idx;
    Query(ll _l, ll _r, ll _i){
        l = _l;
        r = _r;
        idx = _i;
    }

    bool operator<(Query other) const
    {
        return make_pair(l / block_size, r) <
               make_pair(other.l / block_size, other.r);
    }
};

void mo_s_algorithm(vector<Query> &queries, vector<bool> &answers) {
    answers = vector<bool> (queries.size());
    sort(queries.begin(), queries.end());

    odds = 0;    // how many with odd occurences

    ll cur_l = 0;
    ll cur_r = -1;

    for (Query q : queries) {
        if ( (q.r - q.l + 1) % 2){
            answers[q.idx] = false;
            continue;
        }

        while (cur_l > q.l) {
            cur_l--;
            add_el(cur_l);   // add
        }
        while (cur_r < q.r) {
            cur_r++;
            add_el(cur_r);   // add
        }
        while (cur_l < q.l) {
            remove_el(cur_l);   // remove
            cur_l++;
        }
        while (cur_r > q.r) {
            remove_el(cur_r);   // remove
            cur_r--;
        }
        answers[q.idx] = get_answer();
    }
}

// SEGTREE
///ARRAY is 1-indexed
ll T[1000000];  // at least 2*n size (good to go 4*n, because we use 2*n and 2*n+1 stuffs) for segment trees
ll S[1000000];  // lazy

// p is the node/segment
void build(vector<ll> &a, ll p, ll L, ll R){
    //base case
    if (L == R)
        T[p] = a[L];
    else{
        //let us now divide... i.e: for [1, 4] -> [1,2] + [3,4] or [1,5] -> [1,2] + [3,4,5]
        ll mid = (L+R)/2;
        build(a, p*2, L, mid);
        build(a, p*2+1, mid+1, R);

        /// now combine from left and right.. this is the combine function part
        T[p] = min(T[p*2], T[p*2 + 1]);
    }
}

// Point Query
//update node u -> val
void update(ll p, ll u, ll val, ll L, ll R){
    if (L == R){
        T[p] = val;
    }
    else{
        ll mid = (L+R)/2;
        //we are only updating 1 index... so just go the right direction
        if (u <= mid)
            update(p*2, u, val, L, mid);
        else
            update(p*2+1, u, val, mid +1, R);
        T[p] = min(T[p*2], T[p*2 + 1]);
    }
}

ll query(ll p, ll ql, ll qr, ll L, ll R){
    // [ql...qr] is outside [L...R]
    if (ql > qr)
        return INF; //returns INF so, doesn't hamper with other part values

    if (ql == L && qr == R)
        return T[p];

    // push(p); // uncomment IF lazy propagation (range updates)

    ll mid = (L+R)/2;
    ll left = query(p*2, ql, min(qr, mid), L, mid);
    ll right = query(p*2+1, max(ql, mid+1), qr, mid+1, R);
    /// now combine from left and right.. this is the combine function part
    return min(left, right);
}

// lazy push
void push(int p) {
    ll val = S[p];

    //update the children
    T[p*2] += val;
    T[p*2 + 1] += val;

    //pass the lazys to children
    S[p*2] += val;
    S[p*2 + 1] += val;

    //remove the current node lazy
    S[p] = 0;
}

//update [ul,ur] by val
/// keep a lazy at that node
void update(ll p, ll ul, ll ur, ll val, ll L, ll R){
    if (ul > ur)
        return;

    if (ul == L && ur == R){
        S[p] += val;
        T[p] += val;
        return;
    }

    push(p);

    ll mid = (L+R)/2;

    update(p*2, ul, min(ur, mid), val, L, mid);    //update left segment
    update(p*2+1, max(ul, mid+1), ur, val, mid +1, R);  //right segment

    T[p] = max(T[p*2], T[p*2 + 1]); //propagate up
}


// GRAPHS
void djikstra(vector<ll> &dist, ll s, vector<vector<Edge>> adj=front_adj){
    /// s is the source

    priority_queue<PII, vector<PII>, greater<PII>> PQ;
    dist[s] = 0;
    PQ.push({0, s}); //0 distance, 1 is the city

    while (!PQ.empty()){
        PII t = PQ.top();
        PQ.pop();

        if (dist[t.ss] != t.ff) //if not equal, that means this t node is old
            continue;

        for (Edge e : adj[t.ss]){
            if (e.w + t.ff < dist[e.v])
            {
                dist[e.v] = e.w + t.ff;
                PQ.push({dist[e.v],e.v});
            }
        }
    }
}

// BELLMAN FORD
dist = vector<PII>(n+1);

dist[1] = {0, 0};
for (ll i = 2; i <= n; i++)
    dist[i] = {INF, 0};

for (ll i = 1; i <= n; i++){
    for (Edge e: edges){
        if (e.w + dist[e.u].ff < dist[e.v].ff)
            dist[e.v] = {e.w + dist[e.u].ff, e.u}; // u -> v edge
    }
}

bool neg_cyc = false;
ll neg_node = 0;
//check if a negative cycle exists
for (Edge e: edges){
    if (e.w + dist[e.u].ff < dist[e.v].ff){
        dist[e.v] = {e.w + dist[e.u].ff, e.u};
        neg_cyc = true;
        neg_node = e.v;
        //break;
    }
}

// Negative Cycle Path
if (neg_cyc){
    /// IMPORTANT STEP TO NOT GET INTO INFINITE LOOP
    for (ll i = 0; i < n; i++)
        neg_node = dist[neg_node].ss;

    stack<ll> path;
    path.push(neg_node);
    ll tmp_node = dist[neg_node].ss;
    while (tmp_node != neg_node){
        path.push(tmp_node);
        tmp_node = dist[tmp_node].ss;
    }
    path.push(neg_node);
}

// FLOYD-WARSHALL
for (ll k = 1; k <= n; k++){
    for (ll i = 1; i <= n; i++){
        for (ll j = 1; j <= n; j++){
            ll alt_dist = dist[i][k] + dist[k][j];
            if (alt_dist < dist[i][j]){
                dist[i][j] = alt_dist;
                dist[j][i] = alt_dist;
            }
        }
    }
}

// MAXFLOW DINIC
struct FlowEdge {
    int u, v;
    ll cap, flow = 0;
    FlowEdge(int u, int v, ll cap) : u(u), v(v), cap(cap) {}
};

struct Dinic {
    const ll flow_inf = 1e18;
    vector<FlowEdge> edges;
    vector<vector<int>> adj;
    int n, m = 0;
    int s, t;
    vector<int> level, ptr;
    queue<int> q;

    Dinic(int n, int s, int t) : n(n), s(s), t(t) {
        adj.resize(n);
        level.resize(n);
        ptr.resize(n);
    }

    void add_edge(int u, int v, long long cap) {
        edges.emplace_back(u, v, cap);
        edges.emplace_back(v, u, 0);
        adj[u].push_back(m);
        adj[v].push_back(m + 1);
        m += 2;
    }

    bool bfs() {
        while (!q.empty()) {
            int u = q.front();
            q.pop();
            for (int id : adj[u]) {
                // no remaining capacity
                if (edges[id].cap == edges[id].flow)
                    continue;
                // already visited
                if (level[edges[id].v] != -1)
                    continue;
                // node level assigned
                level[edges[id].v] = level[u] + 1;
                q.push(edges[id].v);
            }
        }
        // could sink be reached
        return level[t] != -1;
    }

    // find augmenting paths
    ll dfs(int u, ll pushed) {
        if (pushed == 0)
            return 0;
        if (u == t)
            return pushed;
        for (int& cid = ptr[u]; cid < (int)adj[u].size(); cid++) {
            int id = adj[u][cid];
            int v = edges[id].v;
            // not the forward path (level wise)
            if (level[u] + 1 != level[v])
                continue;

            // recursively... the bottleneck
            ll tr = dfs(v, min(pushed, edges[id].cap - edges[id].flow));
            if (tr == 0)
                continue;

            edges[id].flow += tr;
            edges[id ^ 1].flow -= tr;   // reverse edge
            return tr;
        }
        return 0;
    }

    ll flow() {
        ll f = 0;
        while (true) {
            fill(level.begin(), level.end(), -1);
            level[s] = 0;
            q.push(s);
            if (!bfs())
                break;
            fill(ptr.begin(), ptr.end(), 0);
            while (ll pushed = dfs(s, flow_inf)) {
                f += pushed;
            }
        }
        return f;
    }
};

/*
    Dinic D(n, 0, n-1);
    for (ll i = 0; i < m; i++){
        ll u, v, w;
        cin >> u >> v >> w;
        u--; v--;
        D.add_edge(u, v, w);
    }
    ll ans = D.flow();
*/


// PRIM MST
// MST returns minimum cost (return -1 if not connected)
ll prim(vector<vector<Edge>> adj){

    ll ret = 0;

    priority_queue<PII, vector<PII>, greater<PII>> PQ;


    vector<bool> taken(adj.size());
    vector<ll> cost(adj.size(), INF);
    PQ.push({0, 1}); //0 distance, 1 node
    cost[1] = 0;

    while (!PQ.empty()){
        PII t = PQ.top();
        PQ.pop();

        if (taken[t.ss])
            continue;
        taken[t.ss] = true;
        ret += t.ff;

        for (Edge e : adj[t.ss]){
            // less weight edge to connect e.v
            if (e.w < cost[e.v])
            {
                PQ.push({e.w,e.v});
                cost[e.v] = e.w;    // update the better cost
            }
        }
    }

    for (ll i = 1; i < adj.size(); i++){
        if (cost[i] == INF){
            ret = -1;
            break;
        }
    }
    return ret;
}

// SCC
struct SCC {
    ll N;
    vvl adj, radj;
    vl topo, comp;   // comp is component of node
    vector<bool> vis;

    vvl groups, condG;
    vector<bool> visSCC;
    vl topoSCC;
    void init(ll _N) {
        N = _N;
        adj.resize(N), radj.resize(N), comp = vl(N, -1), vis.resize(N);
    }
    void add_edge(ll u, ll v) { adj[u].pb(v), radj[v].pb(u); }
    void dfs(ll u) {
        vis[u] = 1;
        for (auto &v: adj[u]) if (!vis[v]) dfs(v);
        topo.pb(u);
    }
    void dfs2(ll u, ll p) {
        comp[u] = p;
        groups[p].pb(u);
        for (auto &v: radj[u]) if (comp[v] == -1) dfs2(v, p);
    }

    void dfsSCC(ll u){
        visSCC[u] = 1;
        for (auto &v: condG[u]) if (!visSCC[v]) dfsSCC(v);
        topoSCC.pb(u);
    }

    void gen(ll _N) {  // fills allComp
        rep(i, 1, _N) if (!vis[i]) dfs(i);
        reverse(all(topo));
        
        ll cid = 0;
        for (auto &u: topo) if (comp[u] == -1){
            groups.pb({});
            dfs2(u, cid++);
        }
        
        // build cond. graph
        condG.resize(cid);
        visSCC.resize(cid);
        rep (i, 1, _N) {
            for (auto &j: adj[i]) {
                if (comp[i] == comp[j]) continue;
                    condG[comp[i]].pb(comp[j]); //duplicate edges
            }
        }

        // topOrder of SCC
        rep(i, 1, _N) if (!visSCC[comp[i]]) dfsSCC(comp[i]);
        reverse(all(topoSCC));
    }
};

// MATHS

ll gcd(ll a, ll b){
    if (b == 0)
        return a;
    return gcd(b, a%b);
}

ll power(ll a, ll p, ll m = MOD){
    if (p == 0)
        return 1;
    ll x = power(a, p/2, m);
    x = mul(x,x, m);
    if (p % 2)
        x = mul(a, x, m);
    return x;
}

ll inv_mod(ll a, ll m = MOD){
    return power(a, m-2);
}

// NCR
al<MAXN> fact, invfact;

void initNCR(ll LIM = MAXN) {
    fact[0] = 1;
    ll i;
    for (i = 1; i < MAXN; i++) {
        fact[i] = mul(i, fact[i - 1]);
    }
    i--;
    invfact[i] = inv_mod(fact[i]);
    for (i--; i >= 0; i--) {
        invfact[i] = mul((i + 1), invfact[i + 1]);
    }
}

ll ncr(ll n, ll r) {
    if (r > n || n < 0 || r < 0)
        return 0;
    return mul(mul(fact[n], invfact[r]), invfact[n - r]);
}

// PRIMES & DIVS
vl primes;
map<ll, ll> primeMap;

void sieve(ll LIM = MAXN){
    vector<bool> is_prime(LIM+1, true);

    is_prime[0] = is_prime[1] = false;
    for (ll i = 2; i <= LIM; i++) {
        if (is_prime[i]) {
            primeMap[i] = primes.size();
            primes.push_back(i);

            for (ll j = i * i; j <= LIM; j += i)
                is_prime[j] = false;
        }
    }
}

vl2 to_primes(ll x){
    ll p = primes[0];
    ll i = 1;
    vl2 V;
    while (x != 1 && p * p <= x){
        ll z = 0;
        while (x % p == 0)
        {
            z++;
            x/= p;
        }
        if (z > 0)
            V.push_back({i, z}); //just push index
        p = primes[++i];
    }
    if (x != 1)
        V.push_back({primeMap[x], 1});  // the remaining num itself is prime

    return V;
}

// Need divisors of all numbers 1..n
vector<vector<ll>> divs;
void find_divs(ll LIM = MAXN){
    divs.assign(LIM+1, 0);
    for (ll i = 1; i <= LIM; i++)
        for (ll j = i; j <= LIM; j += i)
            divs[j].push_back(i);
}

// Need divisors of arbitrary numbers repeatedly
// vector<ll> spf;
// void precomputeSPF(ll n = 100000){
//     spf.assign(n+1, 0);
//     for (ll i = 2; i <= n; i++) spf[i] = i;
//     for (ll i = 2; i*i <= n; i++)
//         if (spf[i] == i)
//             for (ll j = i*i; j <= n; j += i)
//                 if (spf[j] == j) spf[j] = i;
// }

// vector<ll> get_divisors(ll x) {
//     vector<pair<ll,ll>> fac;
//     while (x > 1) {
//         ll p = spf[x], cnt = 0;
//         while (x % p == 0) x /= p, cnt++;
//         fac.push_back({p, cnt});
//     }

//     vector<ll> res = {1};
//     for (auto &pr : fac) {
//         ll p = pr.first, c = pr.second;
//         ll sz = res.size();
//         long long mult = 1;
//         for (ll i = 1; i <= c; i++) {
//             mult *= p;
//             for (ll j = 0; j < sz; j++)
//                 res.push_back(res[j] * mult);
//         }
//     }
//     return res;
// }

// TOTIENT

// vector<ll> phi;
// void phi_1_to_n(ll n) {
//     phi.assign(n + 1);
//     for (ll i = 0; i <= n; i++)
//         phi[i] = i;

//     for (ll i = 2; i <= n; i++) {
//         if (phi[i] == i) {
//             for (ll j = i; j <= n; j += i)
//                 phi[j] -= phi[j] / i;
//         }
//     }
// }

// MOBIUS
vector<ll> mu;

void mobius (int lim) {
    mu = vector<ll>(lim);
    vector<int> primes;
    vector<int> lp(lim, 0); // lowest prime factor
    mu[1] = 1;

    for (int i = 2; i < lim; i++) {
        if (lp[i] == 0) {
            lp[i] = i;
            primes.push_back(i);
            mu[i] = -1; // prime → μ(p) = -1
        }

        for (int p : primes) {
            if (p > lp[i] || 1LL * p * i > lim) break;

            lp[p * i] = p;

            if (p == lp[i]) {
                // p divides i → p^2 divides p*i → μ = 0
                mu[p * i] = 0;
                break;
            } else {
                // square-free, one more prime factor → multiply by -1
                mu[p * i] = -mu[i];
            }
        }
    }
}

// STRING
// KMP
void prefix_function(string s, vector<ll> &pi) {
    ll n = s.length();
    pi = vector<ll>(n);
    for (ll i = 1; i < n; i++) {
        ll j = pi[i-1];  ///we try to add up on previous best result...

        ///IF the previous prefix did not work (s[i] != s[j]), let us try with even shorter prefix!!!
        ///and to pick the next probable highest length 'j', we have to pick j = pi[j-1]... THINK!
        while (j > 0 && s[i] != s[j])
            j = pi[j-1];


        ///NOTICE, we add one to prefix value of previous position (we are trying to add one character
                                                                    ///and equal a 'prefix' to suffix)
        //abcabcabgf... at index 6('a').. pi[6] = pi[5] + 1 = 3... as, pi[5] = 2.. and [2] == [6] == c
        //maximum value prefix function can reach is array size(index) = (n-1) (example: 'aaaaaa')
        if (s[i] == s[j])
            j++;

        pi[i] = j;
    }
}

//p = pattern, t = text
///returns a list of occurence positions
void sub_search(string p, string t, vector<ll> &occurs){
    vector<ll> pre;
    prefix_function(p+"#"+t, pre);

    ll k = p.length();
    ll n = k + t.length() + 1;
    for (ll i = k+1; i < n; i++){
        //cout << pre[i] << "a ";

        //i is the end index of occurence (*in the merged text*)
        if (pre[i] == k)
            occurs.push_back(i-2*k);    //pushes start index of occurences
    }
}

/// counts prefixes of s within s or t
void count_prefixes(string s, vector<ll> &cnt){
    vector<ll> pi;
    // calculate the prefix array
    prefix_function(s, pi);

    ll n = s.length();

    cnt = vector<ll> (n + 1); // cnt[i] = prefix of i length

    // at position i, the prefix of length pi[i] ends, so count that
    for (ll i = 0; i < n; i++)  // start from n+1 (if T), i < L (L = cat length)
        cnt[pi[i]]++;
    // for all prefixes of length i,
    // the smaller prefix within that prefix needs to be counted as well
    // the smaller prefix = pi[i-1]  ==> i = length and i-1 = index
    for (ll i = n-1; i > 0; i--)
        cnt[pi[i-1]] += cnt[i];
    // original prefixes (if within T, then no originals)
    for (ll i = 0; i <= n; i++)
        cnt[i]++;
}


// TRIE
class Vertex{
public:
    vector<ll> string_ends; // we will store which strings end
    vector<ll> next;
    ll p;
    char pch;   //edge we took to get here
    ll suffix_link, exit_link;
    vector<ll> go;
    Vertex(ll k, ll pp, char c){
        p = pp;
        pch = c;
        suffix_link = -1;
        exit_link = -1;
        next.resize(k, -1);
        go.resize(k, -1);
    }
};

class Trie{
public:
    ll K;    // K=26 characters in Alphabet
    vector<Vertex> nodes;
    Trie (ll k){
        K = k;
        nodes.push_back(Vertex(K, -1, '#'));
        nodes[0].suffix_link = 0;
    }


    /// THE patterns will be used to build up the trie
    void add(string s, ll idx){
        ll cur = 0;
        for (char ch : s){
            ll c = ch-'a';

            // no path => make path
            if (nodes[cur].next[c] == -1){
                nodes[cur].next[c] = nodes.size();
                nodes.push_back(Vertex(K, cur, ch));
            }
            // travel to next path (whether by making or existing)
            cur = nodes[cur].next[c];
        }
        // end of string (if we only wanted to know if any string ends here
        // we would have kept a boolean like nodes[cur].isWord = true
        nodes[cur].string_ends.push_back(idx);
    }

    // look how many times a string exist
    ll look(string s){
        ll cur = 0;
        for (char ch : s){
            ll c = ch-'a';

            // no path => exit
            if (nodes[cur].next[c] == -1)
                return 0;

            // travel to next path (existing)
            cur = nodes[cur].next[c];
        }
        // end of string
        return nodes[cur].string_ends.size();
    }

    // run full automation that will build up suffix suffix_links and failure paths (go)
    void build_automation(){
        for (ll i = 0; i < nodes.size(); i++){
            //cout << i << endl;
            for (ll j = 0; j < K; j++){

                go(i, j+'a');
            }
        }
    }

    // find all instances of patterns in the string
    void run_automation(string s, vector<ll> &occurs){
        ll cur = 0;
        //cout << cur << " ";
        for (char ch : s){
            ll c = ch-'a';
            cur = nodes[cur].go[c];

            for (ll v : nodes[cur].string_ends)
                occurs[v]++;

            ll e = nodes[cur].exit_link;
            while (e != -1){
                for (ll v : nodes[e].string_ends)
                    occurs[v]++;
                e = nodes[e].exit_link;
            }

            /// prints the order of state visit
            //cout << cur << " ";
        }
        //cout << endl;
    }

    /// also calculates exit links
    ll get_suffix_link(ll v){
        if (nodes[v].suffix_link == -1){

            //root node or parent is root
            if (v == 0  || nodes[v].p == 0){
                nodes[v].suffix_link = 0;
                nodes[v].exit_link = -1;
            }
            else{
                nodes[v].suffix_link = go(get_suffix_link(nodes[v].p), nodes[v].pch);

                //now the exit link
                ll s = nodes[v].suffix_link;
                while (s != 0)
                {
                    if (nodes[s].string_ends.size() > 0){
                        nodes[v].exit_link = s;
                        break;
                    }
                    s = get_suffix_link(s);
                }
            }
        }
        return nodes[v].suffix_link;
    }

    ll go(ll v, char ch){
        ll c = ch-'a';


        if (nodes[v].go[c] == -1){
            //original edge exist
            if (nodes[v].next[c] != -1)
                nodes[v].go[c] = nodes[v].next[c];
            else{
                // if root then itself or thru failure suffix_link (suffix)
                nodes[v].go[c] = (v == 0) ? 0 : go(get_suffix_link(v), ch);
            }
        }
        return nodes[v].go[c];
    }


    void print_trie(ll u, ll t) {
        //spaces
        for (ll s = 0; s < t; s++)
            cout << "  ";

        // with suffix link & exit link
        cout << u << "-> " << nodes[u].pch << " s=" << nodes[u].suffix_link << " e=" << nodes[u].exit_link << " (";
        //cout << u << "-> " << nodes[u].pch << " (";
        for (ll z : nodes[u].string_ends){
            cout << z << " ";
        }
        cout << ")" << endl;
        for (ll i = 0; i < K; i++){
            if (nodes[u].next[i] != -1)
                print_trie(nodes[u].next[i], t+1);
        }
    }
};

/*
Trie t(26);

string text;
cin >> text;

ll n;
cin >> n;

vector<string> patterns(n);


for (ll i = 0; i < n; i++){
    cin >> patterns[i];
    t.add(patterns[i], i);
}


t.build_automation();
t.print_trie(0, 0);

vector<ll> occurs(n, 0);   // which pattern appears how many times
t.run_automation(text, occurs);

for (ll i = 0; i < n; i++){
    cout << occurs[i] << endl;
}
*/


// DP
//1. Order matters + unlimited repetition
for (int x = 0; x <= target; x++)          // sum first
    for (int c : coins)                    // then coins
        if (x >= c)
            dp[x] += dp[x - c];

// 2. Order matters + no repetition
for (int x = target; x >= 0; x--)          // sum backward
    for (int c : coins)                    // coins inner
        if (x >= c)
            dp[x] += dp[x - c];

// 3. Order doesn't matter + unlimited repetition
// STANDARD COIN CHANGE
for (int c : coins)                        // coins first
    for (int x = c; x <= target; x++)      // then sum
        dp[x] += dp[x - c];

// 4. Order doesn't matter + no repetition
// 0-1 KNAPSACK style
for (int c : coins)                        // coins first
    for (int x = target; x >= c; x--)      // sum backward
        dp[x] += dp[x - c];

// Binary Splitting (large k)
for (int i = 0; i < m; i++) {
    int c = coins[i];
    int k = cnt[i];

    // decompose k into sums of powers of 2
    int t = 1;
    while (k > 0) {
        int use = min(t, k);
        int weight = use * c;

        // 0/1 knapsack update
        for (int x = target; x >= weight; x--) {
            dp[x] += dp[x - weight];
        }

        k -= use;
        t <<= 1;
    }
}

/// LCS
string a, b;
cin >> a >> b;
ll A = a.length();
ll B = b.length();
vector<vector<pair<ll, PII> >> dp(A+1, vector<pair<ll, PII>>(B+1, {0, {0, 0}})); // first lcs, second is last common point (for rebuild)
for (ll i = 1; i <= A; i++){
    for (ll j = 1; j <= B; j++){
        if (a[i-1] == b[j-1]){
            dp[i][j] = {dp[i-1][j-1].ff + 1, {i, j}};
        }
        else{
            if (dp[i-1][j].ff > dp[i][j-1].ff)
                dp[i][j] = dp[i-1][j];
            else
                dp[i][j] = dp[i][j-1];
        }
    }
}
string lcs = "";
PII pos = dp[A][B].ss;

while (pos.ff != 0){
    lcs += (char) a[pos.ff-1];  //0-based
    pos.ff--;
    pos.ss--;
    pos = dp[pos.ff][pos.ss].ss;
}
reverse(lcs.begin(), lcs.end());

/// Elevator Rides (max. weight W, min. number number of rides)
vector<PII> dp(1<<n);   // first is min. rides, second is min. weight
dp[0] = {1, 0}; // First elevator
for (ll i = 1; i < (1<<n); i++){
    dp[i] = {INF, INF};

    // j is person (the last person to be included in last ride)
    // try to find without j and consider including j in subset [j belongs to i]
    for (ll j = 0; j < n; j++){
        if (i & (1 << j)){
            // without j
            PII p = dp[i ^ (1 << j)];
            ll s = p.ff;
            ll W = p.ss;

            // needs new lift for "j"
            if (W + w[j] > x){
                s++;
                W = min(W, w[j]);
            }
            else
                W += w[j];

            // less steps OR equal steps but less weight in last lift
            if ((s < dp[i].ff) || (s == dp[i].ff && W < dp[i].ss))
                dp[i] = {s, W};
        }
    }
}

cout << dp[(1<<n)-1].ff << endl;


/// Counting Tiles (SUBMASK DP) => ways to fill n x m with 1x2 / 2x1
ll n, m;
cin >> n >> m;
vector<vector<ll>> dp(2, vector<ll> (1<<n, 0) );
vector<bool> validmasks((1<<n)-1, false);

ll lim = (1<<n)-1;
// base case & valid masks
for (ll i = 0; i < n; i++){
    for (ll s = 0; s <= lim; s++){
        // if 1s are in even segments then valid
        bool valid = true;
        ll c = 0;
        for (ll x = 0; x < n; x++){
            if (s & (1 << x))
                c++;
            else{
                if (c % 2){
                    valid = false;
                    break;
                }
                c = 0;
            }
        }
        if (c % 2)
            valid = false;

        if (valid){
            validmasks[s] = true;
            dp[0][~s & lim] = 1;  // horizontal tiles in those empty places, leaking into next column
        }
    }
}

for (ll j = 1; j < m; j++){
    for (ll s = 0; s <= lim; s++){
        // Column transition: dp[0][s] => dp[1][s]
        // for mask s, the previous column must have 0 where there's 1 (horizontal block used that leaks into next)
        // and in places where 0, it can be 1 or 0
        // So, subsets of ~s
        dp[1][s] = 0;
        ll x = ~s & lim;
        for (ll t = x; ; t=(t-1)&x){
            // t is covered from previous cell (horizontal block)
            // s is covered from this cell (horizontal block)
            if (validmasks[~(t|s) & lim])   // remaining cells can be covered with vertical blocks
                dp[1][s] = (dp[1][s]+dp[0][t])%M;

            if (t == 0)
                break;
        }
    }
    for (ll s = 0; s <= lim; s++)
        dp[0][s] = dp[1][s];
}

cout << dp[0][0] << endl;