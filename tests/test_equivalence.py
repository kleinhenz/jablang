import ablang
import ablang2
import jax
import jablang
import jax.numpy as jnp
import numpy as np
import torch

cpu = jax.devices("cpu")[0]


def test_ablang_heavy():
    m = ablang.pretrained("heavy")
    torch_model = m.AbLang
    torch_model.eval()
    jax_model = jablang.from_torch(torch_model)

    tokens = m.tokenizer(
        [
            "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS"
        ],
        pad=True,
    )

    with torch.no_grad():
        torch_out = torch_model(tokens).numpy()

    with jax.default_device(cpu):
        jax_out = np.array(jax_model(jnp.array(tokens.numpy())))

    diff = np.abs(torch_out - jax_out)
    print(f"ablang heavy: max_abs_err={diff.max():.2e}, mean_abs_err={diff.mean():.2e}")
    np.testing.assert_allclose(torch_out, jax_out, atol=1e-4)


def test_ablang2_paired():
    m = ablang2.pretrained("ablang2-paired")
    torch_model = m.AbLang
    torch_model.eval()
    jax_model = jablang.from_torch(torch_model)

    heavy = "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS"
    light = "DIQMTQSPSSLSASVGDRVTITCRASQGIRNDLGWYQQKPGKAPKLLIYAASSLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCLQHNSYPWTFGQGTKVEIKR"
    tokens = m.tokenizer([(heavy, light)], pad=True)

    with torch.no_grad():
        torch_out = torch_model(tokens).numpy()

    with jax.default_device(cpu):
        jax_out = np.array(jax_model(jnp.array(tokens.numpy())))

    diff = np.abs(torch_out - jax_out)
    print(f"ablang2 paired: max_abs_err={diff.max():.2e}, mean_abs_err={diff.mean():.2e}")
    np.testing.assert_allclose(torch_out, jax_out, atol=1e-4)
