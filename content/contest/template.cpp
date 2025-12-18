#include <bits/stdc++.h>
using namespace std;

template<typename A, typename B> ostream& operator<<(ostream &os, const pair<A, B> &p) { return os << '(' << p.first << ", " << p.second << ')'; }
template<typename T_container, typename T = typename enable_if<!is_same<T_container, string>::value, typename T_container::value_type>::type> ostream& operator<<(ostream &os, const T_container &v) { os << '{'; string sep; for (const T &x : v) os << sep << x, sep = ", "; return os << '}'; }
void dbg_out() { cerr << endl; }
template<typename Head, typename... Tail> void dbg_out(Head H, Tail... T) { cerr << ' ' << H; dbg_out(T...); }
#ifdef LOCAL
#define dbg(...) cerr << __LINE__ << ":" << #__VA_ARGS__, dbg_out(__VA_ARGS__)
#else
#define dbg(...)
#endif

#define rep(i,s,e) for(ll i=(s); i<(e); i++)
#define repr(i,s,e) for(ll i=(s); i>(e); i--)
#define all(a) (a).begin(), (a).end()
#define sz(x) ((int)x.size())
#define pb push_back
#define ff first
#define ss second

using ll = long long;
using dl = double;
using vl = vector<ll>;
using vb = vector<bool>;
template<int SZ> using al = array<ll, SZ>;
using l2 = al<2>;
using vl2 = vector<l2>;

const int MAXN = 1e5 + 5;
const ll MOD = 1e9 + 7;
const ll INF = 1e9;
const dl EPS = 1e-7;

int main() {
    ios::sync_with_stdio(0);cin.tie(0);cin.exceptions(cin.failbit);
    
	ll tcs = 1; 
    cin >> tcs;
    for (ll tc = 1; tc <= tcs; tc++) {
        dbg(tc);
        
        // cout << "Case " << tc << ": " << ans;
        cout << '\n';
    }
    return 0;
}