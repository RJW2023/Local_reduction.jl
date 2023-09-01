function constraint_discretization(f::Function, g::Vector, xL::Vector{Float64}, xU::Vector{Float64}, yL::Vector{Vector{Float64}}, yU::Vector{Vector{Float64}}, N::Int64; solver::Module=Ipopt)
    n_cons = length(SIP_cons) #Number of SIP constranits
    x_size = length(xU)
    y_size = zeros(Int64, n_cons) #Vector of the dimention of variable y in each consraint 
    for i = 1:n_cons
        y_size[i] = length(yU[i])
    end

    y_discre = [zeros(y_size[i], N) for i in 1:n_cons]

    for n = 1:n_cons
        for i = 1:y_size[n]
            y_discre[n][i,:] = collect(range(yL[n][i], stop=yU[n][i], length=N))
        end
    end

    m = Model(solver.Optimizer)
    @variable(m, xL[i] <= x[i=1:x_size] <= xU[i])
    for n = 1:n_cons
        for i = 1:N
            @constraint(m, g[n](x, y_discre[n][:,i])<=0)
        end
    end
    @objective(m, Min, f(x))
    optimize!(m)
    println()
    println("Solution X is $(value.(x))")
end
