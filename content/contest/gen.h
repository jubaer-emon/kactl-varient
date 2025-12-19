/**
 * Author: ju
 * Date: 2025-12-24
 * License: CC0
 * Source: vairous
 * Description: na
 * Status: n/a
 */
#pragma once

using ud = uniform_int_distribution<ll>;
mt19937_64 rng(chrono::steady_clock::now().time_since_epoch().count());

void gen_testcase() {
    ll tcs = ud(1,10)(rng); cout << tcs << "\n";
    rep(tc,0,tcs) {
        ll n = ud(1,100)(rng);
        cout << n << "\n";
        rep(i,0,n) cout << ud(1,1000)(rng) << " ";
        cout << "\n";
    }
    cout << "\n";
}

void gen_tree() {
    ll n = ud(2,20)(rng);
    cout << n << "\n";
    vl perm(n); rep(i,0,n) perm[i] = i+1;
    shuffle(all(perm), rng)
    for (ll v=1; v<n; v++) {
        ll u = ud(0,v-1)(rng); // u=0 -> star, u=v-1 -> line, u=v/2 -> binary tree
        cout << perm[u] << " " << perm[v] << "\n";
    }
}