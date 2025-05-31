function Gkk = Gk(r0, N, k, sf1, sf2)
    w1_ = linspace(-pi,pi,sf1);
    w2_ = linspace(-pi,pi,sf2);
    w1 = repmat(w1_',1,sf2);
    w2 = repmat(w2_,sf1,1);
    absw = sqrt(w1.^2+w2.^2);
    Gkk = exp(1j.*absw.*r0).*Fk(absw,N,k);
end

function F = Fk(w, N, k)
    Wk = pi*(k-1)/(N-1);
    [r, c] = size(w);
    w = w(:);
    d_n = abs(w-Wk);
    d_p = abs(w+Wk);
    F = zeros(length(w),1);
    dp = d_p <= (pi/(N-1));
    dn = d_n <= (pi/(N-1));
    F(dn) = cos(w(dn)-Wk);
    F(dp) = cos(w(dp)+Wk);
    F = reshape(F,r,c);
end